from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# SETTINGS
# ============================================================

# Your dataset
PAGE_IMAGES_DIR = Path(
    "exam_ocr_project/dataset/page_images"
)

# CHANGE THIS
# Example: "student_001"
STUDENT_NAME = "Student_1"

# Output directory
OUTPUT_DIR = Path("splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OCR model
MODEL_NAME = "microsoft/trocr-base-handwritten"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("HANDWRITING OCR TEST")
print("=" * 60)

print(f"Device: {device}")
print(f"Student: {STUDENT_NAME}")


# ============================================================
# STUDENT DIRECTORY
# ============================================================

student_dir = PAGE_IMAGES_DIR / STUDENT_NAME

if not student_dir.exists():
    raise FileNotFoundError(
        f"\nStudent folder not found:\n"
        f"{student_dir.resolve()}\n\n"
        f"Available folders:\n"
        + "\n".join(
            p.name
            for p in PAGE_IMAGES_DIR.iterdir()
            if p.is_dir()
        )
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {
    ".png",
    ".jpg",
    ".jpeg"
}

image_files = sorted(
    [
        p
        for p in student_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]
)

if not image_files:
    raise RuntimeError(
        f"No images found in:\n"
        f"{student_dir.resolve()}"
    )


print(f"\nImages found: {len(image_files)}")

for image in image_files:
    print(f"  - {image.name}")


# ============================================================
# LOAD TROCR
# ============================================================

print("\n" + "=" * 60)
print("LOADING HANDWRITING OCR MODEL")
print("=" * 60)

print("This may take some time the first time.")

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Prepare the handwritten exam page.

    The uploaded pages have:
    - blue handwriting
    - white paper
    - faint text visible from the backside
    - blue page borders
    - horizontal lines
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read: {image_path}"
        )

    # --------------------------------------------------------
    # Resize large images
    # --------------------------------------------------------

    max_width = 1800

    height, width = image.shape[:2]

    if width > max_width:

        scale = max_width / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # BLUE INK MASK
    # --------------------------------------------------------

    lower_blue = np.array(
        [80, 30, 30]
    )

    upper_blue = np.array(
        [140, 255, 255]
    )

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # --------------------------------------------------------
    # DARK PIXELS
    #
    # Useful if some handwriting is not strongly blue.
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = cv2.threshold(
        gray,
        140,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    mask = cv2.bitwise_or(
        blue_mask,
        dark_mask
    )

    # --------------------------------------------------------
    # Remove small noise
    # --------------------------------------------------------

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # REMOVE LEFT PAGE BORDER
    #
    # Your uploaded pages have a vertical blue line here.
    # --------------------------------------------------------

    h, w = mask.shape

    mask[
        :,
        :int(w * 0.06)
    ] = 0

    # --------------------------------------------------------
    # Create black handwriting on white background
    # --------------------------------------------------------

    cleaned = np.full_like(
        gray,
        255
    )

    cleaned[mask > 0] = 0

    return cleaned


# ============================================================
# FIND HANDWRITING LINES
# ============================================================

def detect_lines(binary_image):

    # Count dark pixels per row
    row_pixels = np.sum(
        binary_image < 128,
        axis=1
    )

    # Minimum amount of dark pixels
    threshold = max(
        3,
        int(binary_image.shape[1] * 0.002)
    )

    active = (
        row_pixels > threshold
    )

    ranges = []

    start = None

    # --------------------------------------------------------
    # Find groups of rows containing handwriting
    # --------------------------------------------------------

    for i, value in enumerate(active):

        if value and start is None:

            start = i

        elif not value and start is not None:

            end = i

            if end - start >= 5:

                ranges.append(
                    (start, end)
                )

            start = None

    if start is not None:

        ranges.append(
            (start, len(active))
        )

    # --------------------------------------------------------
    # Merge nearby regions
    # --------------------------------------------------------

    merged = []

    for start, end in ranges:

        if not merged:

            merged.append(
                [start, end]
            )

        else:

            previous = merged[-1]

            if start - previous[1] < 20:

                previous[1] = end

            else:

                merged.append(
                    [start, end]
                )

    return merged


# ============================================================
# OCR ONE LINE
# ============================================================

def recognize_line(line_image):

    # OpenCV grayscale -> RGB
    rgb = cv2.cvtColor(
        line_image,
        cv2.COLOR_GRAY2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    # --------------------------------------------------------
    # Prepare image for TrOCR
    # --------------------------------------------------------

    pixel_values = processor(
        images=pil_image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(
        device
    )

    # --------------------------------------------------------
    # Generate text
    # --------------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# ============================================================
# OCR ONE PAGE
# ============================================================

def process_page(image_path):

    print("\n" + "-" * 60)
    print(f"IMAGE: {image_path.name}")
    print("-" * 60)

    start_time = time.time()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    print("1. Preprocessing image...")

    cleaned = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # DETECT LINES
    # --------------------------------------------------------

    print("2. Detecting handwriting lines...")

    lines = detect_lines(
        cleaned
    )

    print(
        f"   Detected {len(lines)} regions"
    )

    recognized_lines = []

    # --------------------------------------------------------
    # OCR EACH LINE
    # --------------------------------------------------------

    for number, (y1, y2) in enumerate(lines):

        # Padding around line
        padding = 10

        y1 = max(
            0,
            y1 - padding
        )

        y2 = min(
            cleaned.shape[0],
            y2 + padding
        )

        line_image = cleaned[
            y1:y2,
            :
        ]

        if line_image.shape[0] < 10:
            continue

        print(
            f"   OCR line {number + 1}/{len(lines)}..."
        )

        try:

            text = recognize_line(
                line_image
            )

            if text:

                print(
                    f"   → {text}"
                )

                recognized_lines.append(
                    text
                )

        except Exception as e:

            print(
                f"   ERROR: {e}"
            )

    # --------------------------------------------------------
    # COMBINE LINES
    # --------------------------------------------------------

    transcription = " ".join(
        recognized_lines
    )

    elapsed = (
        time.time() - start_time
    )

    print(
        f"\nCompleted in {elapsed:.2f} seconds"
    )

    print("\nFINAL OCR:")
    print(transcription)

    return transcription


# ============================================================
# PROCESS STUDENT
# ============================================================

records = []

print("\n" + "=" * 60)
print("STARTING STUDENT OCR")
print("=" * 60)


for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"\nPAGE {index}/{len(image_files)}"
    )

    try:

        text = process_page(
            image_path
        )

        status = (
            "ok"
            if text
            else "empty"
        )

    except Exception as e:

        print(
            f"\nERROR processing "
            f"{image_path.name}: {e}"
        )

        text = ""
        status = "error"

    records.append(
        {
            "student_name":
                STUDENT_NAME,

            "image_path":
                str(image_path),

            "answer_id":
                index,

            "text":
                text,

            "pen_color":
                "blue",

            "status":
                status
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    records
)

output_file = (
    OUTPUT_DIR /
    f"{STUDENT_NAME}_ocr.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STUDENT OCR FINISHED")
print("=" * 60)

print(
    f"Student: {STUDENT_NAME}"
)

print(
    f"Pages processed: {len(df)}"
)

print(
    f"Successful: "
    f"{(df['status'] == 'ok').sum()}"
)

print(
    f"Empty: "
    f"{(df['status'] == 'empty').sum()}"
)

print(
    f"Errors: "
    f"{(df['status'] == 'error').sum()}"
)

print(
    f"\nCSV saved to:\n"
    f"{output_file.resolve()}"
)

print("\nDone!")