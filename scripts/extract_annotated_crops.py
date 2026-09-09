import os
import json
import cv2

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = "exam_ocr_project"

IMAGE_BASE_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "page_images"
)

ANNOTATION_DIR = os.path.join(
    PROJECT_DIR,
    "annotations"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "annotated_crops"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "annotated_crops_metadata.json"
)

STUDENTS = [
    "Student_1",
    "Student_2",
    "Student_3",
    "Student_4",
    "Student_5",
    "Student_6",
]

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []

print("=" * 70)
print("EXTRACTING MANUALLY ANNOTATED ANSWER CROPS")
print("=" * 70)


# ============================================================
# PROCESS EACH STUDENT
# ============================================================

for student_id in STUDENTS:

    print(f"\nProcessing {student_id}")
    print("-" * 50)

    annotation_file = os.path.join(
        ANNOTATION_DIR,
        f"{student_id}_annotations.json"
    )

    # --------------------------------------------------------
    # CHECK ANNOTATION FILE
    # --------------------------------------------------------

    if not os.path.exists(annotation_file):

        print(" Annotation file not found:")
        print(annotation_file)

        continue

    # --------------------------------------------------------
    # LOAD ANNOTATIONS
    # --------------------------------------------------------

    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # --------------------------------------------------------
    # CREATE STUDENT OUTPUT FOLDER
    # --------------------------------------------------------

    student_output_dir = os.path.join(
        OUTPUT_DIR,
        student_id
    )

    os.makedirs(
        student_output_dir,
        exist_ok=True
    )

    student_crop_count = 0

    # ========================================================
    # PROCESS EACH PAGE
    # ========================================================

    for page_name, questions in annotations.items():

        page_path = os.path.join(
            IMAGE_BASE_DIR,
            student_id,
            page_name
        )

        # ----------------------------------------------------
        # CHECK PAGE IMAGE
        # ----------------------------------------------------

        if not os.path.exists(page_path):

            print(
                f" Page image not found: "
                f"{page_path}"
            )

            continue

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = cv2.imread(page_path)

        if image is None:

            print(
                f" Could not read: "
                f"{page_path}"
            )

            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # PROCESS QUESTIONS
        # ----------------------------------------------------

        for item in questions:

            question_number = item["question_number"]
            question_type = item["type"]
            bbox = item["bbox"]

            x1, y1, x2, y2 = bbox

            # ------------------------------------------------
            # MAKE SURE COORDINATES ARE INSIDE IMAGE
            # ------------------------------------------------

            x1 = max(0, min(x1, image_width))
            x2 = max(0, min(x2, image_width))

            y1 = max(0, min(y1, image_height))
            y2 = max(0, min(y2, image_height))

            # ------------------------------------------------
            # VALIDATE BOUNDING BOX
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:

                print(
                    f" Invalid bbox for "
                    f"Q{question_number} "
                    f"on {page_name}"
                )

                continue

            # ------------------------------------------------
            # CROP ANSWER
            # ------------------------------------------------

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:

                print(
                    f"Empty crop for "
                    f"Q{question_number}"
                )

                continue

            # ------------------------------------------------
            # CREATE FILE NAME
            # ------------------------------------------------

            crop_filename = (
                f"Q{question_number}_"
                f"{question_type}.png"
            )

            crop_path = os.path.join(
                student_output_dir,
                crop_filename
            )

            # ------------------------------------------------
            # SAVE CROP
            # ------------------------------------------------

            cv2.imwrite(
                crop_path,
                crop
            )

            student_crop_count += 1

            # ------------------------------------------------
            # SAVE METADATA
            # ------------------------------------------------

            metadata.append({

                "student_id": student_id,

                "question_number": question_number,

                "question_type": question_type,

                "page": page_name,

                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "image_path": page_path,

                "crop_path": crop_path

            })

            print(
                f"✓ {page_name} | "
                f"Q{question_number} | "
                f"{question_type}"
            )

    print(
        f"\n✓ {student_id}: "
        f"{student_crop_count} crops extracted"
    )


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Total crops extracted: "
    f"{len(metadata)}"
)

print(
    f"\nCrops saved in:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nMetadata saved in:\n"
    f"{METADATA_FILE}"
)