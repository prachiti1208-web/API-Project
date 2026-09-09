
from fastapi import FastAPI
from typing import Optional
import json
import os

app = FastAPI(
    title="Explainable Handwritten Answer Evaluation API",
    description=(
        "API for rubric-grounded handwritten answer evaluation "
        "using Vision Transformers and Semantic NLP."
    ),
    version="1.0"
)

# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "evaluation_report.json"
)


# =========================================================
# LOAD REPORT
# =========================================================

def load_report():

    if not os.path.exists(REPORT_FILE):
        return None

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "project": "Explainable Handwritten Answer Evaluation",
        "status": "API is running",
        "version": "1.0",
        "pipeline": [
            "YOLO Answer Detection",
            "TrOCR Vision Transformer OCR",
            "Semantic NLP",
            "Rubric-Based Evaluation",
            "Explainability",
            "Final Score"
        ]
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Handwritten Answer Evaluation API"
    }


# =========================================================
# POST /evaluate
# =========================================================

@app.post("/evaluate")
def evaluate_student():

    report = load_report()

    if report is None:
        return {
            "status": "error",
            "message": (
                "Evaluation report not found. "
                "Run evaluate_answers.py first."
            )
        }

    # -----------------------------------------------------
    # Basic information
    # -----------------------------------------------------

    student = report.get("student", "Unknown")

    total_score = report.get(
        "total_score",
        report.get("score", 0)
    )

    max_score = report.get(
        "max_score",
        35
    )

    percentage = report.get(
        "percentage",
        round((total_score / max_score) * 100, 2)
        if max_score else 0
    )

    # -----------------------------------------------------
    # Question results
    # -----------------------------------------------------

    question_results = report.get(
        "question_results",
        report.get("results", [])
    )

    # -----------------------------------------------------
    # Calculate summary
    # -----------------------------------------------------

    mcq_correct = 0
    mcq_total = 20

    short_correct = 0
    short_partial = 0
    short_incorrect = 0
    short_total = 15

    for result in question_results:

        question_number = result.get(
            "question_number",
            result.get("question", 0)
        )

        status = str(
            result.get(
                "status",
                result.get("result", "")
            )
        ).lower()

        if question_number <= 20:

            if status == "correct":
                mcq_correct += 1

        else:

            if status == "correct":
                short_correct += 1

            elif status == "partial":
                short_partial += 1

            elif status == "incorrect":
                short_incorrect += 1

    # -----------------------------------------------------
    # Presentation-friendly response
    # -----------------------------------------------------

    return {
        "status": "success",

        "student": {
            "student_id": student
        },

        "final_result": {
            "total_score": total_score,
            "maximum_score": max_score,
            "percentage": percentage,
            "grade_status": (
                "Excellent"
                if percentage >= 85
                else "Good"
                if percentage >= 70
                else "Needs Improvement"
            )
        },

        "section_scores": {

            "mcq": {
                "questions": "Q1-Q20",
                "correct": mcq_correct,
                "total": mcq_total,
                "score": mcq_correct
            },

            "short_answers": {
                "questions": "Q21-Q35",
                "correct": short_correct,
                "partial": short_partial,
                "incorrect": short_incorrect,
                "total": short_total
            }
        },

        "evaluation_method": {
            "mcq": "Exact answer matching",
            "short_answers": [
                "Semantic similarity",
                "Rubric/concept matching",
                "Keyword matching",
                "Explainability analysis"
            ]
        },

        "question_wise_evaluation": question_results
    }


# =========================================================
# GET /results
# =========================================================

@app.get("/results")
def get_results():

    report = load_report()

    if report is None:
        return {
            "status": "error",
            "message": "Evaluation report not found."
        }

    return report


# =========================================================
# GET /results/{student_id}
# =========================================================

@app.get("/results/{student_id}")
def get_student_results(student_id: str):

    report = load_report()

    if report is None:
        return {
            "status": "error",
            "message": "Evaluation report not found."
        }

    report_student = report.get("student", "")

    if report_student.lower() != student_id.lower():

        return {
            "status": "error",
            "message": "Student not found",
            "requested_student": student_id
        }

    return report

