
import os
import json

# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_DIR = os.path.join(
    "exam_ocr_project",
    "annotations"
)

STUDENTS = [
    "Student_1",
    "Student_2",
    "Student_3",
    "Student_4",
    "Student_5",
    "Student_6",
]

MIN_QUESTION = 1
MAX_QUESTION = 35

# ============================================================
# CHECK EACH STUDENT
# ============================================================

all_questions = set()

print("=" * 70)
print("CHECKING MANUAL ANNOTATIONS")
print("=" * 70)

for student in STUDENTS:

    annotation_file = os.path.join(
        ANNOTATION_DIR,
        f"{student}_annotations.json"
    )

    print(f"\n{student}")
    print("-" * 50)

    if not os.path.exists(annotation_file):
        print("❌ Annotation file NOT FOUND")
        print(f"   {annotation_file}")
        continue

    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read JSON: {e}")
        continue

    question_numbers = []

    total_boxes = 0

    # --------------------------------------------------------
    # CHECK PAGES
    # --------------------------------------------------------

    for page_name, annotations in data.items():

        print(f"  {page_name}: {len(annotations)} boxes")

        for item in annotations:

            total_boxes += 1

            q = item.get("question_number")
            qtype = item.get("type")
            bbox = item.get("bbox")

            question_numbers.append(q)
            all_questions.add(q)

            # Check question number
            if not isinstance(q, int):
                print(f"     Invalid question number: {q}")

            elif q < MIN_QUESTION or q > MAX_QUESTION:
                print(f"     Question outside range: Q{q}")

            # Check type
            if q is not None:

                if 1 <= q <= 20:
                    expected_type = "MCQ"
                else:
                    expected_type = "SHORT_ANSWER"

                if qtype != expected_type:
                    print(
                        f"    ⚠️ Q{q}: expected {expected_type}, "
                        f"found {qtype}"
                    )

            # Check bbox
            if not isinstance(bbox, list) or len(bbox) != 4:
                print(f"    ⚠️ Invalid bbox for Q{q}: {bbox}")
                continue

            x1, y1, x2, y2 = bbox

            if x2 <= x1:
                print(f"    ⚠️ Invalid X coordinates for Q{q}")

            if y2 <= y1:
                print(f"    ⚠️ Invalid Y coordinates for Q{q}")

    # --------------------------------------------------------
    # DUPLICATE QUESTIONS
    # --------------------------------------------------------

    duplicates = [
        q for q in set(question_numbers)
        if question_numbers.count(q) > 1
    ]

    if duplicates:
        print(
            f"  ⚠️ Duplicate questions in same student: "
            f"{sorted(duplicates)}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(f"  Total boxes: {total_boxes}")
    print(f"  Questions found: {sorted(question_numbers)}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print(
    "Questions appearing anywhere in the 6 students:"
)
print(sorted(all_questions))

print("\nExpected question range:")
print("Q1–Q35")

print("\n Annotation validation finished.")

