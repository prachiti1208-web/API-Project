"""
STEP 6
------

TrOCR recognition on YOLO-generated SHORT_ANSWER crops
for Students 7-50.

YOLO has already been trained using Students 1-6 and
applied to Students 7-50.

Example YOLO crop:

    page_1_answer_001_SHORT_ANSWER.png

Important:
    We DO NOT assume that answer_001 == Q21.

    YOLO may detect:
    - multiple regions belonging to one answer
    - extra regions
    - answers spanning multiple regions

Therefore this step only performs OCR and preserves:
    - student
    - page
    - crop number
    - crop path
    - OCR text

Question mapping will be handled separately.
"""

import os
import re
import json
import torch

from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Current project:
# C:\Users\jpras\OneDrive\Documents\exam_ocr_project\
#     exam_ocr_project

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)


# Existing YOLO output
INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "answer_detection",
    "students_7_to_50"
)


# TrOCR output
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ocr_results",
    "short_answers"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

MODEL_NAME = "microsoft/trocr-base-handwritten"

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("STEP 6 - TrOCR SHORT ANSWER RECOGNITION")
print("=" * 70)

print()
print("Device:", DEVICE)
print("Input :", INPUT_DIR)
print("Output:", OUTPUT_DIR)


# ============================================================
# 4. CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_DIR):

    print()
    print("ERROR: YOLO output directory does not exist:")
    print(INPUT_DIR)

    raise SystemExit


# ============================================================
# 5. LOAD TrOCR
# ============================================================

print()
print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("TrOCR loaded successfully.")


# ============================================================
# 6. OCR FUNCTION
# ============================================================

def recognize_image(image_path):

    """
    Run TrOCR on one YOLO crop.
    """

    image = Image.open(
        image_path
    ).convert("RGB")


    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values


    pixel_values = pixel_values.to(
        DEVICE
    )


    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=256
        )


    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]


    return text.strip()


# ============================================================
# 7. PARSE YOLO FILENAME
# ============================================================

def parse_crop_filename(filename):

    """
    Parse filenames such as:

        page_1_answer_001_SHORT_ANSWER.png

    Returns:

        page = 1
        crop_number = 1
        type = SHORT_ANSWER
    """

    pattern = (
        r"page_(\d+)"
        r"_answer_(\d+)"
        r"_(SHORT_ANSWER|MCQ)"
        r"\.(png|jpg|jpeg|webp)$"
    )


    match = re.match(
        pattern,
        filename,
        flags=re.IGNORECASE
    )


    if not match:

        return None


    page_number = int(
        match.group(1)
    )

    crop_number = int(
        match.group(2)
    )

    answer_type = match.group(3).upper()


    return {
        "page": page_number,
        "crop_number": crop_number,
        "type": answer_type
    }


# ============================================================
# 8. GET STUDENT DIRECTORIES
# ============================================================

student_dirs = []

for item in os.listdir(INPUT_DIR):

    full_path = os.path.join(
        INPUT_DIR,
        item
    )

    if os.path.isdir(full_path):

        student_dirs.append(
            full_path
        )


# Sort Student_7, Student_8, ... Student_50
def student_sort_key(path):

    name = os.path.basename(path)

    match = re.search(
        r"(\d+)",
        name
    )

    if match:

        return int(
            match.group(1)
        )

    return 999


student_dirs.sort(
    key=student_sort_key
)


print()
print(
    "Student folders found:",
    len(student_dirs)
)


# ============================================================
# 9. PROCESS EACH STUDENT
# ============================================================

for student_dir in student_dirs:

    student_name = os.path.basename(
        student_dir
    )


    print()
    print("=" * 70)
    print("PROCESSING:", student_name)
    print("=" * 70)


    # --------------------------------------------------------
    # Find crops directory
    # --------------------------------------------------------

    crops_dir = os.path.join(
        student_dir,
        "crops"
    )


    if not os.path.exists(crops_dir):

        print(
            "WARNING: crops folder not found:"
        )

        print(crops_dir)

        continue


    # --------------------------------------------------------
    # Find all image files
    # --------------------------------------------------------

    image_files = []

    for filename in os.listdir(crops_dir):

        if filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):

            image_files.append(
                os.path.join(
                    crops_dir,
                    filename
                )
            )


    image_files.sort()


    # --------------------------------------------------------
    # Keep ONLY short-answer crops
    # --------------------------------------------------------

    short_answer_files = []

    for image_path in image_files:

        filename = os.path.basename(
            image_path
        )


        if "SHORT_ANSWER" in filename.upper():

            short_answer_files.append(
                image_path
            )


    print(
        "Short-answer crops:",
        len(short_answer_files)
    )


    # --------------------------------------------------------
    # OCR results
    # --------------------------------------------------------

    short_answers = []


    for index, image_path in enumerate(
        short_answer_files,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )


        metadata = parse_crop_filename(
            filename
        )


        if metadata is None:

            print(
                f"[{index}/{len(short_answer_files)}]"
                f" Invalid filename: {filename}"
            )

            continue


        print()
        print(
            f"[{index}/{len(short_answer_files)}]"
        )

        print(
            "Crop:",
            filename
        )

        print(
            "Page:",
            metadata["page"]
        )

        print(
            "Crop number:",
            metadata["crop_number"]
        )


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        try:

            text = recognize_image(
                image_path
            )


            print(
                "OCR:",
                text
            )


            short_answers.append({

                "page":
                    metadata["page"],

                "crop_number":
                    metadata["crop_number"],

                "type":
                    "SHORT_ANSWER",

                "crop_filename":
                    filename,

                "crop_path":
                    os.path.abspath(
                        image_path
                    ),

                "text":
                    text

            })


        except Exception as error:

            print(
                "OCR ERROR:",
                str(error)
            )


            short_answers.append({

                "page":
                    metadata["page"],

                "crop_number":
                    metadata["crop_number"],

                "type":
                    "SHORT_ANSWER",

                "crop_filename":
                    filename,

                "crop_path":
                    os.path.abspath(
                        image_path
                    ),

                "text":
                    "",

                "error":
                    str(error)

            })


    # ========================================================
    # SAVE JSON
    # ========================================================

    output_data = {

        "student":
            student_name,

        "description":
            "TrOCR recognition of YOLO-generated "
            "short-answer regions.",

        "question_mapping":
            "NOT_ASSIGNED",

        "short_answer_regions":
            short_answers

    }


    output_file = os.path.join(
        OUTPUT_DIR,
        f"{student_name}.json"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )


    print()
    print(
        "Saved:",
        output_file
    )


# ============================================================
# 10. COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)

print()
print("TrOCR results saved to:")

print(OUTPUT_DIR)

print()
print(
    "IMPORTANT: Question numbers have NOT been assigned yet."
)

print(
    "The OCR JSON preserves the original YOLO page/crop"
)

print(
    "information so we can perform question mapping next."
)