import os
import json
import cv2

# ============================================================
# SETTINGS
# ============================================================

STUDENT_ID = "Student_6"

IMAGE_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "page_images",
    STUDENT_ID
)

ANNOTATION_FILE = os.path.join(
    "exam_ocr_project",
    "annotations",
    f"{STUDENT_ID}_annotations.json"
)

OUTPUT_DIR = os.path.join(
    "exam_ocr_project",
    "annotations",
    "visualized",
    STUDENT_ID
)

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD ANNOTATIONS
# ============================================================

if not os.path.exists(ANNOTATION_FILE):
    print("ERROR: Annotation file not found:")
    print(ANNOTATION_FILE)
    exit()

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# ============================================================
# PROCESS EACH PAGE
# ============================================================

for page_name, boxes in annotations.items():

    image_path = os.path.join(IMAGE_DIR, page_name)

    if not os.path.exists(image_path):
        print(f"WARNING: Image not found: {image_path}")
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    # Draw every annotation
    for item in boxes:

        question_number = item["question_number"]
        question_type = item["type"]

        x1, y1, x2, y2 = item["bbox"]

        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

        # Label
        label = f"Q{question_number} - {question_type}"

        # Label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            5
        )

        cv2.rectangle(
            image,
            (x1, max(0, y1 - text_height - 20)),
            (x1 + text_width + 20, y1),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            image,
            label,
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            5,
            cv2.LINE_AA
        )

    # ========================================================
    # SAVE VISUALIZED IMAGE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        page_name
    )

    cv2.imwrite(output_path, image)

    print(f"Saved: {output_path}")

print("\n========================================")
print("VISUALIZATION COMPLETE")
print("========================================")
print(f"Open this folder:")
print(OUTPUT_DIR)