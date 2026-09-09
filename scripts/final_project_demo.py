import os
import json
import re
from datetime import datetime

# ============================================================
# FINAL PROJECT DEMO
# ============================================================
#
# Project:
# Explainable Rubric-Grounded Handwritten Answer Evaluation
# Using Vision Transformers and Semantic NLP
#
# PURPOSE:
#   1. Read OCR output for Student_7
#   2. Show YOLO/TrOCR pipeline information
#   3. Use a controlled structured answer set for evaluation
#   4. Run semantic + concept/rubric evaluation
#   5. Generate a complete final report
#
# IMPORTANT:
# The current TrOCR output is noisy.
# Therefore, this demo DOES NOT falsely map noisy OCR fragments
# to Q21-Q35.
#
# The evaluation module is demonstrated using the validated
# structured student-answer input.
#
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

OCR_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "ocr_results",
    "short_answers",
    "Student_7.json"
)

ANSWER_KEY_FILE = os.path.join(
    BASE_DIR,
    "metadata",
    "answerkey.txt"
)

STUDENT_ANSWERS_FILE = os.path.join(
    BASE_DIR,
    "student_answers.json"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

REPORT_DIR = os.path.join(
    BASE_DIR,
    "final_demo_results"
)

os.makedirs(REPORT_DIR, exist_ok=True)

FINAL_REPORT = os.path.join(
    REPORT_DIR,
    "Student_7_Final_Project_Report.json"
)

SUMMARY_REPORT = os.path.join(
    REPORT_DIR,
    "Student_7_Summary.txt"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize(text):
    """Normalize text for comparison."""

    if text is None:
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


def load_answer_key():

    answer_key = {}

    if not os.path.exists(ANSWER_KEY_FILE):

        print("[ERROR] Answer key not found:")
        print(ANSWER_KEY_FILE)

        return answer_key

    with open(
        ANSWER_KEY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("Question_Number"):
            continue

        parts = line.split(",", 2)

        if len(parts) < 3:
            continue

        q_no = parts[0].strip()

        q_type = parts[1].strip()

        answer = parts[2].strip()

        answer = answer.strip('"')

        answer_key[q_no] = {
            "type": q_type,
            "answer": answer
        }

    return answer_key


def load_student_answers():

    if not os.path.exists(STUDENT_ANSWERS_FILE):

        print("[ERROR] student_answers.json not found:")
        print(STUDENT_ANSWERS_FILE)

        return None

    with open(
        STUDENT_ANSWERS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


def inspect_ocr():

    result = {
        "available": False,
        "student": "Student_7",
        "regions": 0,
        "pages": [],
        "ocr_status": "Not available"
    }

    if not os.path.exists(OCR_FILE):

        return result

    try:

        with open(
            OCR_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        regions = data.get(
            "short_answer_regions",
            []
        )

        pages = sorted(
            list(
                set(
                    str(r.get("page"))
                    for r in regions
                )
            )
        )

        result = {
            "available": True,
            "student": data.get(
                "student",
                "Student_7"
            ),
            "regions": len(regions),
            "pages": pages,
            "ocr_status":
                "Available but noisy; "
                "not used for automatic question mapping"
        }

    except Exception as e:

        result["ocr_status"] = (
            "OCR file found but could not be parsed: "
            + str(e)
        )

    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate_mcq(
    q_no,
    correct_answer,
    student_answer
):

    correct = normalize(correct_answer)

    student = normalize(student_answer)

    if student == correct:

        return {
            "status": "Correct",
            "score": 1.0
        }

    return {
        "status": "Incorrect",
        "score": 0.0
    }


def evaluate_short_answer(
    q_no,
    correct_answer,
    student_answer
):

    # --------------------------------------------------------
    # This function provides a lightweight explainable
    # demonstration.
    #
    # The main validated evaluator remains:
    # evaluate_answers.py
    #
    # Therefore this final demo reads its existing report
    # when available.
    # --------------------------------------------------------

    return {
        "status": "Evaluated by semantic/rubric engine",
        "score": None
    }


def load_existing_evaluation():

    evaluation_file = os.path.join(
        BASE_DIR,
        "evaluation_report.json"
    )

    if not os.path.exists(evaluation_file):

        return None

    try:

        with open(
            evaluation_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# BUILD FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("FINAL PROJECT DEMONSTRATION")
print("=" * 72)

print()
print("Project:")
print(
    "Explainable Rubric-Grounded Handwritten "
    "Answer Evaluation Using Vision Transformers "
    "and Semantic NLP"
)

print()
print("-" * 72)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

answer_key = load_answer_key()

student_data = load_student_answers()

ocr_info = inspect_ocr()

existing_evaluation = load_existing_evaluation()


if student_data is None:

    raise SystemExit(
        "\nstudent_answers.json is required for the "
        "validated evaluation demonstration."
    )


student_name = student_data.get(
    "student",
    "Student_7"
)


# ============================================================
# PIPELINE INFORMATION
# ============================================================

pipeline = {

    "stage_1_dataset":
        "Handwritten student answer sheets",

    "stage_2_detection":
        "YOLO answer-region detection",

    "stage_3_recognition":
        "TrOCR Vision Transformer handwriting recognition",

    "stage_4_answer_representation":
        "Structured student-answer JSON",

    "stage_5_evaluation":
        "Semantic similarity + concept/rubric matching",

    "stage_6_explanation":
        "Matched concepts, missing concepts and score",

    "stage_7_output":
        "Explainable evaluation report"
}


# ============================================================
# EVALUATION RESULTS
# ============================================================

if existing_evaluation:

    evaluation = existing_evaluation

else:

    print(
        "\n[WARNING] evaluation_report.json was not found."
    )

    print(
        "Run evaluate_answers.py once before this demo."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# Extract summary safely
# ------------------------------------------------------------

total_score = evaluation.get(
    "total_score",
    evaluation.get("score", 0)
)

total_questions = evaluation.get(
    "total_questions",
    35
)

percentage = evaluation.get(
    "percentage",
    0
)


# Try different possible field names used by evaluator
mcq_correct = evaluation.get(
    "mcq_correct",
    evaluation.get(
        "mcq",
        {}
    ).get("correct", 0)
    if isinstance(evaluation.get("mcq"), dict)
    else 0
)

short_correct = evaluation.get(
    "short_correct",
    evaluation.get(
        "short_answers",
        {}
    ).get("correct", 0)
    if isinstance(evaluation.get("short_answers"), dict)
    else 0
)

short_partial = evaluation.get(
    "short_partial",
    evaluation.get(
        "short_answers",
        {}
    ).get("partial", 0)
    if isinstance(evaluation.get("short_answers"), dict)
    else 0
)

short_incorrect = evaluation.get(
    "short_incorrect",
    evaluation.get(
        "short_answers",
        {}
    ).get("incorrect", 0)
    if isinstance(evaluation.get("short_answers"), dict)
    else 0
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 72)
print("FINAL EVALUATION RESULT")
print("=" * 72)

print()
print(f"Student       : {student_name}")
print(f"Total Score   : {total_score}/{total_questions}")
print(f"Percentage    : {percentage:.2f}%")

print()
print("MCQ SECTION")
print("-" * 72)
print(f"Questions     : Q1-Q20")
print(f"Correct       : {mcq_correct}/20")

print()
print("SHORT ANSWER SECTION")
print("-" * 72)
print(f"Questions     : Q21-Q35")
print(f"Correct       : {short_correct}/15")
print(f"Partial       : {short_partial}/15")
print(f"Incorrect     : {short_incorrect}/15")


# ============================================================
# OCR INFORMATION
# ============================================================

print()
print("=" * 72)
print("OCR / DETECTION INFORMATION")
print("=" * 72)

print()

if ocr_info["available"]:

    print("Student             :", ocr_info["student"])
    print("OCR regions         :", ocr_info["regions"])
    print(
        "Pages processed     :",
        ", ".join(ocr_info["pages"])
    )

    print(
        "OCR status          :",
        ocr_info["ocr_status"]
    )

else:

    print("OCR output not found.")


# ============================================================
# CREATE FINAL REPORT
# ============================================================

final_report = {

    "project": {
        "title":
            "Explainable Rubric-Grounded Handwritten "
            "Answer Evaluation Using Vision Transformers "
            "and Semantic NLP",

        "prototype_status":
            "Working evaluation prototype",

        "generated_at":
            datetime.now().isoformat()
    },

    "student": student_name,

    "pipeline": pipeline,

    "detection_and_ocr": ocr_info,

    "evaluation": evaluation,

    "methodology": {

        "mcq_evaluation":
            "Exact normalized answer matching",

        "short_answer_evaluation":
            "Sentence Transformer semantic similarity "
            "combined with concept/keyword rubric matching",

        "explainability":
            "Reports matched concepts, missing concepts, "
            "semantic similarity and evaluation decision"
    },

    "important_note":
        "The current YOLO/TrOCR output contains noisy "
        "recognition and unreliable question mapping. "
        "Therefore the validated structured student-answer "
        "input is used for the evaluation demonstration. "
        "This prevents the prototype from falsely claiming "
        "that noisy OCR text was correctly mapped."
}


# ============================================================
# SAVE JSON REPORT
# ============================================================

with open(
    FINAL_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_report,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# CREATE HUMAN-READABLE SUMMARY
# ============================================================

with open(
    SUMMARY_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPLAINABLE HANDWRITTEN ANSWER "
        "EVALUATION PROJECT\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "Project: Explainable Rubric-Grounded "
        "Handwritten Answer Evaluation Using "
        "Vision Transformers and Semantic NLP\n\n"
    )

    f.write(
        f"Student: {student_name}\n"
    )

    f.write(
        f"Score: {total_score}/{total_questions}\n"
    )

    f.write(
        f"Percentage: {percentage:.2f}%\n\n"
    )

    f.write(
        "PIPELINE\n"
    )

    f.write("-" * 65 + "\n")

    for key, value in pipeline.items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write("\n")

    f.write(
        "EVALUATION METHOD\n"
    )

    f.write("-" * 65 + "\n")

    f.write(
        "Q1-Q20: Exact normalized MCQ matching\n"
    )

    f.write(
        "Q21-Q35: Semantic similarity + "
        "concept/keyword rubric matching\n"
    )

    f.write(
        "Explainability: matched concepts, "
        "missing concepts and similarity scores\n"
    )

    f.write("\n")

    f.write(
        "OCR STATUS\n"
    )

    f.write("-" * 65 + "\n")

    f.write(
        f"Detected regions: "
        f"{ocr_info.get('regions', 0)}\n"
    )

    f.write(
        f"Status: "
        f"{ocr_info.get('ocr_status', 'Unknown')}\n"
    )

    f.write("\n")

    f.write(
        "NOTE\n"
    )

    f.write("-" * 65 + "\n")

    f.write(
        "The current OCR output is noisy and question "
        "mapping is not reliable. The evaluation module "
        "was therefore validated using structured "
        "student-answer data rather than falsely claiming "
        "perfect OCR recognition.\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 72)
print("DEMO COMPLETE")
print("=" * 72)

print()
print("Final JSON report:")
print(FINAL_REPORT)

print()
print("Human-readable summary:")
print(SUMMARY_REPORT)

print()
print("Your prototype pipeline is ready for demonstration.")
print("=" * 72)