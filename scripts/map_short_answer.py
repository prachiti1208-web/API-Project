"""
STEP 7
------

Create question mapping templates for SHORT_ANSWER regions.

Input:
    dataset/ocr_results/short_answers/

Output:
    dataset/ocr_results/mapped_short_answers/

Important:
    We DO NOT assume:
        answer_001 == Q21
        answer_002 == Q22
        etc.

This step preserves the detected OCR regions and creates
Q21-Q35 mapping slots.

If a student has more/fewer than 15 detected regions,
the student is flagged for review.

No OCR is performed in this step.
"""

import os
import json
import re


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ocr_results",
    "short_answers"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ocr_results",
    "mapped_short_answers"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# QUESTION RANGE
# ============================================================

START_QUESTION = 21
END_QUESTION = 35

QUESTIONS = list(range(START_QUESTION, END_QUESTION + 1))


# ============================================================
# SORT FUNCTION
# ============================================================

def region_sort_key(region):

    page = region.get("page", 999)

    crop_number = region.get("crop_number", 999)

    return (page, crop_number)


# ============================================================
# CREATE MAPPING FOR ONE STUDENT
# ============================================================

def create_mapping(data):

    student = data.get("student", "Unknown")

    regions = data.get("short_answer_regions", [])

    # Sort regions according to page and crop number
    regions = sorted(
        regions,
        key=region_sort_key
    )

    # --------------------------------------------------------
    # Create empty Q21-Q35 structure
    # --------------------------------------------------------

    answers = []

    for q in QUESTIONS:

        answers.append({
            "question_number": q,

            # Region filenames belonging to this question
            "regions": [],

            # Combined OCR text
            "student_answer": "",

            # Mapping status
            "mapping_status": "NOT_MAPPED"
        })


    # --------------------------------------------------------
    # Determine whether number of regions is exactly 15
    # --------------------------------------------------------

    region_count = len(regions)

    if region_count == 15:

        status = "REVIEW_REQUIRED"

    elif region_count > 15:

        status = "MULTIPLE_REGIONS_OR_EXTRA_DETECTIONS"

    elif region_count < 15:

        status = "MISSING_REGIONS"

    else:

        status = "UNKNOWN"


    # --------------------------------------------------------
    # Preserve all detected regions separately
    # --------------------------------------------------------

    detected_regions = []

    for region in regions:

        detected_regions.append({

            "page": region.get("page"),

            "crop_number": region.get("crop_number"),

            "type": region.get("type"),

            "crop_filename": region.get("crop_filename"),

            "crop_path": region.get("crop_path"),

            "ocr_text": region.get("text", "")
        })


    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output = {

        "student": student,

        "step": 7,

        "description":
            "Question mapping template for Q21-Q35. "
            "No automatic assumption that crop number equals question number.",

        "total_detected_regions": region_count,

        "expected_questions": 15,

        "overall_status": status,

        "questions": answers,

        "detected_regions": detected_regions
    }

    return output


# ============================================================
# PROCESS ALL STUDENTS
# ============================================================

def main():

    print("=" * 70)
    print("STEP 7 - SHORT ANSWER QUESTION MAPPING")
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_DIR)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()

    if not os.path.exists(INPUT_DIR):

        print("ERROR: Input directory does not exist.")
        print(INPUT_DIR)
        return


    json_files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".json")
    ]

    json_files.sort()


    if not json_files:

        print("ERROR: No JSON files found.")
        return


    print("Students found:", len(json_files))
    print()


    summary = []


    for filename in json_files:

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        try:

            with open(
                input_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)


            mapped_data = create_mapping(data)


            student = mapped_data["student"]

            region_count = mapped_data[
                "total_detected_regions"
            ]

            status = mapped_data[
                "overall_status"
            ]


            # Output filename
            output_filename = (
                student + "_mapped.json"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )


            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    mapped_data,
                    f,
                    indent=4,
                    ensure_ascii=False
                )


            summary.append({

                "student": student,

                "regions": region_count,

                "status": status
            })


            print(
                f"{student}: "
                f"{region_count} regions -> "
                f"{status}"
            )


        except Exception as e:

            print(
                f"ERROR processing {filename}: {e}"
            )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "mapping_summary.json"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7 COMPLETED")
    print("=" * 70)

    print()

    print(
        "Mapped template files:",
        len(json_files)
    )

    print()

    print(
        "Output folder:"
    )

    print(OUTPUT_DIR)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These files are mapping templates."
    )

    print(
        "They do NOT assume answer_001 = Q21."
    )

    print(
        "Students with != 15 regions need special grouping."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()