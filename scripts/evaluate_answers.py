"""
evaluate_answers.py

Explainable Rubric-Grounded Handwritten Answer Evaluation

Q1-Q20  -> Exact MCQ matching
Q21-Q35 -> Semantic similarity + concept matching + keyword overlap

Input:
    metadata/answerkey.txt
    student_answers.json

Output:
    evaluation_report.json

Required:
    pip install sentence-transformers
"""

import os
import re
import json
from typing import Dict, List, Tuple


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ANSWER_KEY_PATH = os.path.join(
    PROJECT_ROOT,
    "metadata",
    "answerkey.txt"
)

STUDENT_ANSWERS_PATH = os.path.join(
    PROJECT_ROOT,
    "student_answers.json"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation_report.json"
)


# ============================================================
# SCORING
# ============================================================

CORRECT_SCORE = 1.0
PARTIAL_SCORE = 0.5
INCORRECT_SCORE = 0.0

SEMANTIC_CORRECT_THRESHOLD = 0.68
SEMANTIC_PARTIAL_THRESHOLD = 0.45

CONCEPT_CORRECT_THRESHOLD = 0.65
CONCEPT_PARTIAL_THRESHOLD = 0.35

COMBINED_CORRECT_THRESHOLD = 0.62
COMBINED_PARTIAL_THRESHOLD = 0.40


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

try:
    from sentence_transformers import SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True

except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("/", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace(":", " ")

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(text: str) -> List[str]:

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


# ============================================================
# LOAD ANSWER KEY
# ============================================================

def load_answer_key(
    path: str
) -> Dict[int, Dict[str, str]]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Answer key not found:\n{path}"
        )

    answer_key = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.lower().startswith(
            "question_number"
        ):
            continue

        parts = []
        current = ""
        inside_quotes = False

        for char in line:

            if char == '"':
                inside_quotes = not inside_quotes
                continue

            if char == "," and not inside_quotes:

                parts.append(
                    current.strip()
                )

                current = ""

            else:

                current += char

        parts.append(
            current.strip()
        )

        if len(parts) < 3:
            continue

        try:

            question_number = int(
                parts[0]
            )

        except ValueError:

            continue

        question_type = parts[1]

        correct_answer = ",".join(
            parts[2:]
        ).strip()

        answer_key[question_number] = {
            "type": question_type,
            "answer": correct_answer
        }

    return answer_key


# ============================================================
# LOAD STUDENT ANSWERS
# ============================================================

def load_student_answers(
    path: str
) -> Tuple[str, Dict[int, str]]:

    """
    Supports BOTH formats.

    FORMAT 1:

    {
        "student": "Student_7",
        "answers": [
            {
                "question_number": 1,
                "student_answer": "B"
            }
        ]
    }

    FORMAT 2:

    {
        "student": "Student_7",
        "answers": {
            "1": "B",
            "2": "C"
        }
    }
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Student answers file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):

        raise ValueError(
            "student_answers.json must contain a JSON object."
        )

    student_name = data.get(
        "student",
        "Unknown"
    )

    raw_answers = data.get(
        "answers",
        []
    )

    answers = {}

    # --------------------------------------------------------
    # FORMAT 1: LIST
    # --------------------------------------------------------

    if isinstance(raw_answers, list):

        for item in raw_answers:

            if not isinstance(
                item,
                dict
            ):
                continue

            question_number = item.get(
                "question_number"
            )

            student_answer = item.get(
                "student_answer",
                ""
            )

            if question_number is None:
                continue

            try:

                question_number = int(
                    question_number
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            answers[
                question_number
            ] = str(
                student_answer
            )

    # --------------------------------------------------------
    # FORMAT 2: DICTIONARY
    # --------------------------------------------------------

    elif isinstance(raw_answers, dict):

        for question_number, student_answer in raw_answers.items():

            try:

                question_number = int(
                    question_number
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            # Supports:

            # "1": "B"

            # OR

            # "1": {
            #     "student_answer": "B"
            # }

            if isinstance(
                student_answer,
                dict
            ):

                student_answer = student_answer.get(
                    "student_answer",
                    ""
                )

            answers[
                question_number
            ] = str(
                student_answer
            )

    else:

        raise ValueError(
            "Invalid 'answers' format. "
            "Expected a list or dictionary."
        )

    return student_name, answers


# ============================================================
# RUBRIC CONCEPTS
# ============================================================

CONCEPTS = {

    21: {
        "supervised learning": [
            "supervised learning",
            "supervised"
        ],
        "labeled data": [
            "labeled data",
            "labelled data",
            "labeled",
            "labelled"
        ],
        "inputs and outputs": [
            "inputs and outputs",
            "input and output",
            "known inputs",
            "known outputs"
        ]
    },

    22: {
        "overfitting": [
            "overfitting",
            "overfit"
        ],
        "memorized training data": [
            "memorized training data",
            "memorized the training data",
            "memorizes training data",
            "memorised training data"
        ],
        "generalization": [
            "failed to generalize",
            "cannot generalize",
            "does not generalize",
            "poor generalization",
            "new data"
        ]
    },

    23: {
        "regression": [
            "regression"
        ],
        "statistical method": [
            "statistical method",
            "statistical process",
            "statistical"
        ],
        "dependent variable": [
            "dependent variable"
        ],
        "independent variable": [
            "independent variable",
            "independent variables"
        ],
        "relationship": [
            "relationship",
            "relationships"
        ]
    },

    24: {
        "missing values": [
            "missing values",
            "missing value"
        ],
        "data quality": [
            "data quality",
            "poor data quality"
        ],
        "overfitting": [
            "overfitting",
            "overfit"
        ],
        "underfitting": [
            "underfitting",
            "underfit"
        ],
        "data collection": [
            "data collection"
        ],
        "time consumption": [
            "time consumption",
            "time consuming"
        ]
    },

    25: {
        "healthcare": [
            "healthcare",
            "health care"
        ],
        "disease prediction": [
            "disease prediction",
            "disease"
        ],
        "finance": [
            "finance",
            "financial"
        ],
        "fraud detection": [
            "fraud detection",
            "fraud"
        ],
        "e-commerce": [
            "e-commerce",
            "ecommerce",
            "e commerce"
        ],
        "recommendation systems": [
            "recommendation systems",
            "recommendation",
            "recommendations"
        ]
    },

    26: {
        "structured data": [
            "structured data",
            "structured"
        ],
        "rows and columns": [
            "rows and columns",
            "row and column"
        ],
        "SQL": [
            "sql"
        ],
        "Excel": [
            "excel"
        ],
        "semi-structured data": [
            "semi-structured data",
            "semi structured data",
            "semi-structured",
            "semi structured"
        ],
        "JSON": [
            "json"
        ],
        "XML": [
            "xml"
        ],
        "HTML": [
            "html"
        ]
    },

    27: {
        "Z-score": [
            "z-score",
            "z score",
            "zscore"
        ],
        "IQR": [
            "iqr",
            "interquartile range"
        ],
        "scatter plots": [
            "scatter plots",
            "scatter plot",
            "scatter"
        ],
        "box plots": [
            "box plots",
            "box plot",
            "boxplot"
        ],
        "trimming": [
            "trimming",
            "trim"
        ],
        "capping": [
            "capping",
            "cap"
        ],
        "imputation": [
            "imputation",
            "impute"
        ]
    },

    28: {
        "data preprocessing": [
            "data preprocessing",
            "preprocessing",
            "pre processing"
        ],
        "cleaning": [
            "clean",
            "cleaning"
        ],
        "transforming": [
            "transforming",
            "transform",
            "transformation"
        ],
        "raw data": [
            "raw data"
        ],
        "machine learning": [
            "machine learning",
            "ml model"
        ]
    },

    29: {
        "K-NN": [
            "k-nn",
            "knn",
            "k nn"
        ],
        "distance based": [
            "distance-based",
            "distance based",
            "distance"
        ],
        "larger scales dominate": [
            "larger scales",
            "larger scale",
            "dominate"
        ],
        "normalization": [
            "normalized",
            "normalization",
            "normalize",
            "scaling"
        ]
    },

    30: {
        "binning": [
            "binning",
            "bins",
            "bin"
        ],
        "grouping values": [
            "grouping values",
            "groups values",
            "group values",
            "grouping"
        ],
        "intervals": [
            "intervals",
            "interval"
        ],
        "age categories": [
            "ages",
            "age"
        ],
        "child adult senior": [
            "child",
            "adult",
            "senior"
        ]
    },

    31: {
        "number of clusters": [
            "number of clusters",
            "number of cluster",
            "clusters"
        ],
        "groups": [
            "groups",
            "group"
        ],
        "predefined": [
            "predefined",
            "pre defined"
        ]
    },

    32: {
        "learning rate": [
            "learning rate"
        ],
        "number of trees": [
            "number of trees",
            "trees",
            "tree"
        ],
        "max depth": [
            "max depth",
            "maximum depth"
        ],
        "batch size": [
            "batch size",
            "batch"
        ],
        "K in K-NN": [
            "k in k-nn",
            "k in knn",
            "k-nn",
            "knn"
        ]
    },

    33: {
        "gradient descent": [
            "gradient descent"
        ],
        "optimization algorithm": [
            "optimization algorithm",
            "optimization"
        ],
        "cost function": [
            "cost function",
            "minimize cost"
        ],
        "iterative movement": [
            "iteratively",
            "iteration",
            "iterative",
            "moving toward"
        ],
        "steepest descent": [
            "steepest descent"
        ]
    },

    34: {
        "SVM": [
            "svm",
            "support vector machine"
        ],
        "maximize margin": [
            "maximize the margin",
            "maximizing the margin",
            "maximize margin",
            "maximum margin",
            "maximizing margin",
            "margin"
        ],
        "gutter": [
            "gutter"
        ],
        "data points outside margin": [
            "outside the margin",
            "outside margin",
            "no data points fall within",
            "data points outside",
            "points outside"
        ]
    },

    35: {
        "higher dimensional space": [
            "higher-dimensional space",
            "higher dimensional space",
            "higher dimension"
        ],
        "nonlinear decision boundary": [
            "non-linear decision boundary",
            "nonlinear decision boundary",
            "non linear decision boundary",
            "nonlinear boundary",
            "decision boundary"
        ],
        "transformation": [
            "transforms data",
            "transform data",
            "transformed",
            "transform"
        ]
    }
}


# ============================================================
# CONCEPT MATCHING
# ============================================================

def concept_is_present(
    student_text: str,
    patterns: List[str]
) -> bool:

    normalized_student = normalize_text(
        student_text
    )

    for pattern in patterns:

        normalized_pattern = normalize_text(
            pattern
        )

        if not normalized_pattern:
            continue

        if normalized_pattern in normalized_student:
            return True

        pattern_tokens = normalized_pattern.split()

        if len(pattern_tokens) == 1:

            if re.search(
                r"\b"
                + re.escape(pattern_tokens[0])
                + r"\b",
                normalized_student
            ):

                return True

    return False


def calculate_concept_score(
    question_number: int,
    student_answer: str
) -> Tuple[float, List[str], List[str]]:

    concepts = CONCEPTS.get(
        question_number,
        {}
    )

    if not concepts:

        return 0.0, [], []

    matched_concepts = []
    missing_concepts = []

    for concept_name, patterns in concepts.items():

        if concept_is_present(
            student_answer,
            patterns
        ):

            matched_concepts.append(
                concept_name
            )

        else:

            missing_concepts.append(
                concept_name
            )

    concept_score = (
        len(matched_concepts)
        /
        len(concepts)
    )

    return (
        concept_score,
        matched_concepts,
        missing_concepts
    )


# ============================================================
# QUESTION-SPECIFIC COMPLETENESS
# ============================================================

def calculate_completeness(
    question_number: int,
    student_answer: str
) -> float:

    text = normalize_text(
        student_answer
    )

    # --------------------------------------------------------
    # Q32 - Hyperparameters
    # --------------------------------------------------------

    if question_number == 32:

        valid_items = [
            "learning rate",
            "trees",
            "number of trees",
            "max depth",
            "maximum depth",
            "batch size",
            "k-nn",
            "knn"
        ]

        found = set()

        for item in valid_items:

            normalized_item = normalize_text(
                item
            )

            if normalized_item in text:
                found.add(
                    normalized_item
                )

        if len(found) >= 3:
            return 1.0

        if len(found) == 2:
            return 0.70

        if len(found) == 1:
            return 0.40

        return 0.0

    # --------------------------------------------------------
    # Q34 - SVM margin
    # --------------------------------------------------------

    if question_number == 34:

        margin = any(
            phrase in text
            for phrase in [
                "maximize the margin",
                "maximizing the margin",
                "maximize margin",
                "maximum margin",
                "margin"
            ]
        )

        points = any(
            phrase in text
            for phrase in [
                "data points",
                "data point",
                "points"
            ]
        )

        gutter_or_position = any(
            phrase in text
            for phrase in [
                "gutter",
                "outside the margin",
                "outside margin",
                "within the margin"
            ]
        )

        if (
            margin
            and points
            and gutter_or_position
        ):
            return 1.0

        if margin and points:
            return 0.85

        if margin:
            return 0.65

        return 0.0

    return 0.0


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

def load_semantic_model():

    if not SENTENCE_TRANSFORMERS_AVAILABLE:

        print(
            "[WARNING] sentence-transformers is not installed."
        )

        print(
            "[WARNING] Using lexical fallback."
        )

        return None

    print(
        "[INFO] Loading Sentence Transformer model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "[INFO] Sentence Transformer loaded successfully."
    )

    return model


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    model,
    student_answer: str,
    correct_answer: str
) -> float:

    if not student_answer.strip():
        return 0.0

    if not correct_answer.strip():
        return 0.0

    if model is None:

        student_tokens = set(
            tokenize(student_answer)
        )

        correct_tokens = set(
            tokenize(correct_answer)
        )

        if not student_tokens:
            return 0.0

        if not correct_tokens:
            return 0.0

        intersection = (
            student_tokens &
            correct_tokens
        )

        union = (
            student_tokens |
            correct_tokens
        )

        if not union:
            return 0.0

        return (
            len(intersection)
            /
            len(union)
        )

    embeddings = model.encode(
        [
            student_answer,
            correct_answer
        ],
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return similarity


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def calculate_keyword_overlap(
    student_answer: str,
    correct_answer: str
) -> float:

    student_tokens = set(
        tokenize(student_answer)
    )

    reference_tokens = set(
        tokenize(correct_answer)
    )

    if not student_tokens:
        return 0.0

    if not reference_tokens:
        return 0.0

    common = (
        student_tokens &
        reference_tokens
    )

    return (
        len(common)
        /
        len(reference_tokens)
    )


# ============================================================
# SHORT ANSWER EVALUATION
# ============================================================

def evaluate_short_answer(
    question_number: int,
    student_answer: str,
    correct_answer: str,
    model
) -> Dict:

    semantic_score = calculate_semantic_similarity(
        model,
        student_answer,
        correct_answer
    )

    concept_score, matched_concepts, missing_concepts = (
        calculate_concept_score(
            question_number,
            student_answer
        )
    )

    keyword_score = calculate_keyword_overlap(
        student_answer,
        correct_answer
    )

    completeness_score = calculate_completeness(
        question_number,
        student_answer
    )

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    combined_score = (
        0.55 * semantic_score
        +
        0.30 * concept_score
        +
        0.15 * keyword_score
    )

    if completeness_score > 0:

        combined_score = max(
            combined_score,
            (
                0.65 * combined_score
                +
                0.35 * completeness_score
            )
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if (
        combined_score >= COMBINED_CORRECT_THRESHOLD
        or
        (
            semantic_score >= SEMANTIC_CORRECT_THRESHOLD
            and
            concept_score >= CONCEPT_PARTIAL_THRESHOLD
        )
        or
        (
            concept_score >= CONCEPT_CORRECT_THRESHOLD
            and
            semantic_score >= SEMANTIC_PARTIAL_THRESHOLD
        )
    ):

        status = "Correct"
        score = CORRECT_SCORE

    elif (
        combined_score >= COMBINED_PARTIAL_THRESHOLD
        or
        semantic_score >= SEMANTIC_PARTIAL_THRESHOLD
        or
        concept_score >= CONCEPT_PARTIAL_THRESHOLD
    ):

        status = "Partial"
        score = PARTIAL_SCORE

    else:

        status = "Incorrect"
        score = INCORRECT_SCORE

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    explanation_parts = []

    if matched_concepts:

        explanation_parts.append(
            "Matched concepts: "
            + ", ".join(
                matched_concepts
            )
        )

    if missing_concepts:

        explanation_parts.append(
            "Missing concepts: "
            + ", ".join(
                missing_concepts
            )
        )

    if semantic_score >= SEMANTIC_CORRECT_THRESHOLD:

        explanation_parts.append(
            "High semantic similarity."
        )

    elif semantic_score >= SEMANTIC_PARTIAL_THRESHOLD:

        explanation_parts.append(
            "Moderate semantic similarity."
        )

    else:

        explanation_parts.append(
            "Low semantic similarity."
        )

    explanation = " ".join(
        explanation_parts
    )

    return {

        "question_number":
            question_number,

        "type":
            "Short_Answer",

        "student_answer":
            student_answer,

        "correct_answer":
            correct_answer,

        "status":
            status,

        "score":
            score,

        "semantic_similarity":
            round(
                semantic_score,
                4
            ),

        "concept_score":
            round(
                concept_score,
                4
            ),

        "keyword_overlap":
            round(
                keyword_score,
                4
            ),

        "matched_concepts":
            matched_concepts,

        "missing_concepts":
            missing_concepts,

        "explanation":
            explanation
    }


# ============================================================
# MCQ EVALUATION
# ============================================================

def evaluate_mcq(
    question_number: int,
    student_answer: str,
    correct_answer: str
) -> Dict:

    student_normalized = normalize_text(
        student_answer
    )

    correct_normalized = normalize_text(
        correct_answer
    )

    if (
        student_normalized
        ==
        correct_normalized
    ):

        status = "Correct"
        score = CORRECT_SCORE

    else:

        status = "Incorrect"
        score = INCORRECT_SCORE

    return {

        "question_number":
            question_number,

        "type":
            "MCQ",

        "student_answer":
            student_answer,

        "correct_answer":
            correct_answer,

        "status":
            status,

        "score":
            score
    }


# ============================================================
# MAIN
# ============================================================

def evaluate_student():

    print("=" * 70)

    print(
        "EXPLAINABLE RUBRIC-GROUNDED ANSWER EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # ANSWER KEY
    # --------------------------------------------------------

    print(
        f"[INFO] Loading answer key:\n"
        f"{ANSWER_KEY_PATH}"
    )

    answer_key = load_answer_key(
        ANSWER_KEY_PATH
    )

    print(
        f"[INFO] Loaded {len(answer_key)} questions."
    )

    # --------------------------------------------------------
    # STUDENT ANSWERS
    # --------------------------------------------------------

    print(
        f"[INFO] Loading student answers:\n"
        f"{STUDENT_ANSWERS_PATH}"
    )

    student_name, student_answers = (
        load_student_answers(
            STUDENT_ANSWERS_PATH
        )
    )

    print(
        f"[INFO] Student: {student_name}"
    )

    print(
        f"[INFO] Loaded {len(student_answers)} student answers."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_semantic_model()

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    results = []

    mcq_correct = 0
    mcq_incorrect = 0
    mcq_score = 0.0

    short_correct = 0
    short_partial = 0
    short_incorrect = 0
    short_score = 0.0

    # --------------------------------------------------------
    # EVALUATE ALL QUESTIONS
    # --------------------------------------------------------

    for question_number in sorted(
        answer_key.keys()
    ):

        question_info = answer_key[
            question_number
        ]

        question_type = question_info[
            "type"
        ]

        correct_answer = question_info[
            "answer"
        ]

        student_answer = student_answers.get(
            question_number,
            ""
        )

        # ====================================================
        # MCQ
        # ====================================================

        if question_type.upper() == "MCQ":

            result = evaluate_mcq(
                question_number,
                student_answer,
                correct_answer
            )

            results.append(
                result
            )

            if result["status"] == "Correct":

                mcq_correct += 1
                mcq_score += result["score"]

            else:

                mcq_incorrect += 1

        # ====================================================
        # SHORT ANSWER
        # ====================================================

        else:

            result = evaluate_short_answer(
                question_number,
                student_answer,
                correct_answer,
                model
            )

            results.append(
                result
            )

            if result["status"] == "Correct":

                short_correct += 1

            elif result["status"] == "Partial":

                short_partial += 1

            else:

                short_incorrect += 1

            short_score += result["score"]

    # ========================================================
    # FINAL SCORE
    # ========================================================

    total_score = (
        mcq_score
        +
        short_score
    )

    maximum_score = len(
        answer_key
    )

    percentage = (
        total_score
        /
        maximum_score
        *
        100
        if maximum_score > 0
        else 0
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "student":
            student_name,

        "project":
            "Explainable Rubric-Grounded Handwritten Answer Evaluation",

        "summary": {

            "total_score":
                round(
                    total_score,
                    2
                ),

            "maximum_score":
                maximum_score,

            "percentage":
                round(
                    percentage,
                    2
                ),

            "mcq": {

                "question_range":
                    "Q1-Q20",

                "total_questions":
                    20,

                "correct":
                    mcq_correct,

                "incorrect":
                    mcq_incorrect,

                "score":
                    round(
                        mcq_score,
                        2
                    ),

                "maximum_score":
                    20
            },

            "short_answers": {

                "question_range":
                    "Q21-Q35",

                "total_questions":
                    15,

                "correct":
                    short_correct,

                "partial":
                    short_partial,

                "incorrect":
                    short_incorrect,

                "score":
                    round(
                        short_score,
                        2
                    ),

                "maximum_score":
                    15
            }
        },

        "evaluation_method": {

            "mcq":
                "Exact normalized answer matching",

            "short_answer":
                "Sentence Transformer semantic similarity + rubric concept matching + keyword overlap",

            "semantic_model":
                (
                    "all-MiniLM-L6-v2"
                    if SENTENCE_TRANSFORMERS_AVAILABLE
                    else "Lexical fallback"
                ),

            "semantic_correct_threshold":
                SEMANTIC_CORRECT_THRESHOLD,

            "semantic_partial_threshold":
                SEMANTIC_PARTIAL_THRESHOLD,

            "combined_correct_threshold":
                COMBINED_CORRECT_THRESHOLD,

            "combined_partial_threshold":
                COMBINED_PARTIAL_THRESHOLD,

            "concept_matching":
                "Question-specific rubric concepts"
        },

        "results":
            results
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Student       : {student_name}"
    )

    print(
        f"Total Score   : "
        f"{total_score:.1f}/{maximum_score}"
    )

    print(
        f"Percentage    : "
        f"{percentage:.2f}%"
    )

    print()

    print(
        "MCQ Q1-Q20"
    )

    print(
        f"Correct       : "
        f"{mcq_correct}/20"
    )

    print(
        f"Incorrect     : "
        f"{mcq_incorrect}/20"
    )

    print(
        f"Score         : "
        f"{mcq_score:.1f}/20"
    )

    print()

    print(
        "Short Answers Q21-Q35"
    )

    print(
        f"Correct       : "
        f"{short_correct}/15"
    )

    print(
        f"Partial       : "
        f"{short_partial}/15"
    )

    print(
        f"Incorrect     : "
        f"{short_incorrect}/15"
    )

    print(
        f"Score         : "
        f"{short_score:.1f}/15"
    )

    print()

    print(
        f"[INFO] Report saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)

    # ========================================================
    # EXPLANATIONS
    # ========================================================

    print()
    print(
        "SHORT ANSWER EXPLANATIONS"
    )

    print("=" * 70)

    for result in results:

        if result["type"] != "Short_Answer":
            continue

        print()

        print(
            f"Q{result['question_number']}: "
            f"{result['status']} "
            f"({result['score']}/1)"
        )

        print(
            f"  Semantic similarity : "
            f"{result['semantic_similarity']}"
        )

        print(
            f"  Concept score       : "
            f"{result['concept_score']}"
        )

        print(
            f"  Keyword overlap     : "
            f"{result['keyword_overlap']}"
        )

        if result["matched_concepts"]:

            print(
                "  Matched concepts    : "
                + ", ".join(
                    result["matched_concepts"]
                )
            )

        if result["missing_concepts"]:

            print(
                "  Missing concepts    : "
                + ", ".join(
                    result["missing_concepts"]
                )
            )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        evaluate_student()

    except FileNotFoundError as error:

        print()
        print(
            "[ERROR] File not found:"
        )

        print(error)

    except json.JSONDecodeError as error:

        print()
        print(
            "[ERROR] Invalid JSON:"
        )

        print(error)

    except Exception as error:

        print()
        print(
            "[ERROR] Evaluation failed:"
        )

        print(error)

        raise