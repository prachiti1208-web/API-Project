"""
ROBUST QUESTION-LEVEL DETECTION AND CROPPING
================================================

TEST VERSION:
    Processes ONLY Student_1.

Main strategy:
    1. Read page images.
    2. Detect text regions using EasyOCR.
    3. Identify candidate question numbers.
    4. Validate candidates using:
          - OCR pattern
          - question range
          - position
          - confidence
          - sequence consistency
    5. Remove duplicate question anchors.
    6. Create question-level regions.
    7. Handle questions continuing across pages.
    8. Save debug images.
    9. Save detection results as JSON.

IMPORTANT:
    This version does NOT assume:
        - Q1-Q20 are on page 1
        - Q21-Q35 are on page 2
        - every question is answered
        - every student has identical page layouts
"""

import os
import re
import json
import cv2
import numpy as np
import easyocr


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = "Student_1"

PAGE_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "page_images",
    STUDENT_ID
)

OUTPUT_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "question_crops",
    STUDENT_ID
)

DEBUG_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "question_debug",
    STUDENT_ID
)

MIN_QUESTION = 1
MAX_QUESTION = 35

DETECTION_WIDTH = 1500

PAD_X = 35
PAD_Y = 25

MIN_REGION_WIDTH = 10
MIN_REGION_HEIGHT = 8

QUESTION_NUMBER_CONFIDENCE = 0.30

# Maximum reasonable distance between duplicate
# detections of the same question number.
DUPLICATE_DISTANCE = 120

# Question number should usually be toward the
# left side of the page.
LEFT_SIDE_RATIO = 0.40


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


# ============================================================
# LOAD EASYOCR
# ============================================================

print("\nLoading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded.\n")


# ============================================================
# NORMALIZE OCR TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Remove spaces
    text = re.sub(r"\s+", "", text)

    return text


# ============================================================
# EXTRACT QUESTION NUMBER
# ============================================================

def extract_question_number(text):

    """
    Detect question numbers such as:

        1
        1.
        1)
        Q1
        Q.1
        q1.
        21
        21.
        21)

    Returns:
        integer
        or None
    """

    if not text:
        return None

    text = normalize_text(text)

    # --------------------------------------------------------
    # Remove obvious surrounding noise
    # --------------------------------------------------------

    text = text.strip()

    # --------------------------------------------------------
    # Q1 / Q.1 / q1.
    # --------------------------------------------------------

    pattern_q = r"^q\.?(\d{1,2})[\.\):\-]?$"

    match = re.fullmatch(
        pattern_q,
        text
    )

    if match:

        number = int(match.group(1))

        if MIN_QUESTION <= number <= MAX_QUESTION:
            return number

    # --------------------------------------------------------
    # Pure number
    # --------------------------------------------------------

    pattern_number = r"^(\d{1,2})[\.\):\-]?$"

    match = re.fullmatch(
        pattern_number,
        text
    )

    if match:

        number = int(match.group(1))

        if MIN_QUESTION <= number <= MAX_QUESTION:
            return number

    return None


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_for_detection(image):

    h, w = image.shape[:2]

    if w <= DETECTION_WIDTH:

        return image.copy(), 1.0, 1.0

    new_w = DETECTION_WIDTH

    new_h = int(
        h * new_w / w
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    scale_x = w / new_w
    scale_y = h / new_h

    return resized, scale_x, scale_y


# ============================================================
# DETECT TEXT REGIONS
# ============================================================

def detect_regions(image):

    resized, scale_x, scale_y = resize_for_detection(
        image
    )

    results = reader.readtext(
        resized,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        height_ths=0.5,
        ycenter_ths=0.5,
        mag_ratio=1.0
    )

    regions = []

    for item in results:

        if len(item) != 3:
            continue

        bbox, text, confidence = item

        xs = [
            point[0]
            for point in bbox
        ]

        ys = [
            point[1]
            for point in bbox
        ]

        x1 = int(
            min(xs) * scale_x
        )

        y1 = int(
            min(ys) * scale_y
        )

        x2 = int(
            max(xs) * scale_x
        )

        y2 = int(
            max(ys) * scale_y
        )

        width = x2 - x1
        height = y2 - y1

        if width < MIN_REGION_WIDTH:
            continue

        if height < MIN_REGION_HEIGHT:
            continue

        regions.append({

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],

            "text": str(text).strip(),

            "confidence": float(
                confidence
            )
        })

    return regions


# ============================================================
# QUESTION CANDIDATE VALIDATION
# ============================================================

def validate_question_candidate(
    region,
    page_width
):

    text = region["text"]

    confidence = region["confidence"]

    bbox = region["bbox"]

    x1, y1, x2, y2 = bbox

    number = extract_question_number(
        text
    )

    # --------------------------------------------------------
    # Rule 1: must contain valid question number
    # --------------------------------------------------------

    if number is None:
        return None

    # --------------------------------------------------------
    # Rule 2: confidence
    # --------------------------------------------------------

    if confidence < QUESTION_NUMBER_CONFIDENCE:
        return None

    # --------------------------------------------------------
    # Rule 3: question number should generally
    # be toward the left side.
    #
    # NOTE:
    # This is not a hard requirement for the entire
    # project. It is a useful validation signal.
    # --------------------------------------------------------

    center_x = (
        x1 + x2
    ) / 2

    left_side = (
        center_x <
        page_width * LEFT_SIDE_RATIO
    )

    # --------------------------------------------------------
    # Calculate score rather than immediately rejecting.
    # --------------------------------------------------------

    score = 0

    # Valid number
    score += 2

    # OCR confidence
    if confidence >= 0.70:
        score += 3

    elif confidence >= 0.50:
        score += 2

    else:
        score += 1

    # Left side
    if left_side:
        score += 2

    # Very small text boxes are more likely to be
    # question labels.
    width = x2 - x1
    height = y2 - y1

    if width < page_width * 0.10:
        score += 1

    if height < 150:
        score += 1

    return {

        "question_number": number,

        "bbox": [
            x1,
            y1,
            x2,
            y2
        ],

        "text": text,

        "confidence": confidence,

        "candidate_score": score,

        "left_side": left_side
    }


# ============================================================
# FIND QUESTION ANCHORS
# ============================================================

def find_question_anchors(
    regions,
    page_width
):

    candidates = []

    for region in regions:

        candidate = validate_question_candidate(
            region,
            page_width
        )

        if candidate is not None:

            candidates.append(
                candidate
            )

    return candidates


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_anchors(
    anchors
):

    if not anchors:
        return []

    cleaned = []

    for anchor in anchors:

        x1, y1, x2, y2 = anchor["bbox"]

        cx = (
            x1 + x2
        ) / 2

        cy = (
            y1 + y2
        ) / 2

        duplicate_index = None

        for i, existing in enumerate(cleaned):

            if (
                anchor["question_number"]
                !=
                existing["question_number"]
            ):
                continue

            ex1, ey1, ex2, ey2 = (
                existing["bbox"]
            )

            ecx = (
                ex1 + ex2
            ) / 2

            ecy = (
                ey1 + ey2
            ) / 2

            distance = np.sqrt(
                (cx - ecx) ** 2 +
                (cy - ecy) ** 2
            )

            if distance < DUPLICATE_DISTANCE:

                duplicate_index = i
                break

        if duplicate_index is None:

            cleaned.append(
                anchor
            )

        else:

            existing = cleaned[
                duplicate_index
            ]

            if (
                anchor["candidate_score"]
                >
                existing["candidate_score"]
            ):

                cleaned[
                    duplicate_index
                ] = anchor

    return cleaned


# ============================================================
# SORT ANCHORS
# ============================================================

def sort_anchors(anchors):

    return sorted(
        anchors,
        key=lambda a: (
            a["bbox"][1],
            a["bbox"][0]
        )
    )


# ============================================================
# CHECK QUESTION SEQUENCE
# ============================================================

def calculate_sequence_score(
    anchors
):

    """
    Questions normally increase:

        Q1 → Q2 → Q3 → Q4

    This function does not require perfect sequence,
    because unanswered questions are allowed.

    It simply helps identify suspicious detections.
    """

    if len(anchors) < 2:
        return anchors

    previous = None

    for anchor in anchors:

        qnum = anchor[
            "question_number"
        ]

        if previous is None:

            anchor[
                "sequence_status"
            ] = "FIRST"

        elif qnum > previous:

            anchor[
                "sequence_status"
            ] = "OK"

        elif qnum == previous:

            anchor[
                "sequence_status"
            ] = "DUPLICATE"

        else:

            anchor[
                "sequence_status"
            ] = "OUT_OF_ORDER"

        previous = qnum

    return anchors


# ============================================================
# CREATE DEBUG IMAGE
# ============================================================

def draw_debug_image(
    image,
    regions,
    anchors
):

    debug = image.copy()

    # --------------------------------------------------------
    # Draw all OCR regions
    # --------------------------------------------------------

    for region in regions:

        x1, y1, x2, y2 = (
            region["bbox"]
        )

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            1
        )

        text = region[
            "text"
        ][:25]

        cv2.putText(
            debug,
            text,
            (
                x1,
                max(
                    15,
                    y1 - 3
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1
        )

    # --------------------------------------------------------
    # Draw question anchors
    # --------------------------------------------------------

    for anchor in anchors:

        x1, y1, x2, y2 = (
            anchor["bbox"]
        )

        qnum = anchor[
            "question_number"
        ]

        score = anchor[
            "candidate_score"
        ]

        cv2.rectangle(
            debug,
            (
                max(0, x1 - 8),
                max(0, y1 - 8)
            ),
            (
                x2 + 8,
                y2 + 8
            ),
            (0, 0, 255),
            4
        )

        label = (
            f"Q{qnum} "
            f"S={score}"
        )

        cv2.putText(
            debug,
            label,
            (
                x1,
                max(
                    35,
                    y1 - 12
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return debug


# ============================================================
# CREATE SINGLE-PAGE QUESTION CROP
# ============================================================

def create_single_page_crop(
    image,
    current_anchor,
    next_anchor=None
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = (
        current_anchor["bbox"]
    )

    crop_top = max(
        0,
        y1 - PAD_Y
    )

    # --------------------------------------------------------
    # If another question exists on same page
    # --------------------------------------------------------

    if next_anchor is not None:

        nx1, ny1, nx2, ny2 = (
            next_anchor["bbox"]
        )

        crop_bottom = max(
            crop_top + 50,
            ny1 - PAD_Y
        )

    else:

        crop_bottom = h

    # --------------------------------------------------------
    # Use most of the page width.
    #
    # This is intentional because handwritten answers
    # can extend far to the right.
    # --------------------------------------------------------

    crop_left = 0

    crop_right = w

    crop = image[
        crop_top:crop_bottom,
        crop_left:crop_right
    ]

    return crop


# ============================================================
# PROCESS PAGE
# ============================================================

def process_page(
    page_path,
    page_number
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING PAGE {page_number}"
    )

    print("=" * 70)

    image = cv2.imread(
        page_path
    )

    if image is None:

        print(
            "ERROR: Could not read:",
            page_path
        )

        return {
            "page": page_number,
            "regions": [],
            "anchors": []
        }

    h, w = image.shape[:2]

    print(
        f"Image size: {w} x {h}"
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    regions = detect_regions(
        image
    )

    print(
        f"Detected OCR regions: "
        f"{len(regions)}"
    )

    # --------------------------------------------------------
    # Find candidates
    # --------------------------------------------------------

    anchors = find_question_anchors(
        regions,
        w
    )

    print(
        f"Question candidates: "
        f"{len(anchors)}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    anchors = remove_duplicate_anchors(
        anchors
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    anchors = sort_anchors(
        anchors
    )

    # --------------------------------------------------------
    # Sequence analysis
    # --------------------------------------------------------

    anchors = calculate_sequence_score(
        anchors
    )

    # --------------------------------------------------------
    # Print anchors
    # --------------------------------------------------------

    print("\nDetected anchors:")

    for anchor in anchors:

        print(
            f"Q{anchor['question_number']} "
            f"| text='{anchor['text']}' "
            f"| confidence="
            f"{anchor['confidence']:.2f} "
            f"| score="
            f"{anchor['candidate_score']} "
            f"| y="
            f"{anchor['bbox'][1]} "
            f"| sequence="
            f"{anchor['sequence_status']}"
        )

    # --------------------------------------------------------
    # Debug image
    # --------------------------------------------------------

    debug = draw_debug_image(
        image,
        regions,
        anchors
    )

    debug_path = os.path.join(
        DEBUG_DIR,
        f"page_{page_number}_debug.png"
    )

    cv2.imwrite(
        debug_path,
        debug
    )

    print(
        f"\nDebug saved:"
        f"\n{debug_path}"
    )

    return {

        "page": page_number,

        "image_path": page_path,

        "width": w,

        "height": h,

        "regions": regions,

        "anchors": anchors
    }


# ============================================================
# CREATE QUESTION CROPS ACROSS PAGES
# ============================================================

def create_question_crops(
    page_results
):

    """
    Create question crops.

    IMPORTANT:
    A question can continue onto the next page.

    Example:

        Page 2:
            Q24
            answer...
            answer...

        Page 3:
            continued answer...
            Q25

    In that case, Q24 spans:

        Page 2 + beginning of Page 3
    """

    question_records = []

    # --------------------------------------------------------
    # Flatten all anchors into page-aware list
    # --------------------------------------------------------

    all_anchors = []

    for page_result in page_results:

        page = page_result[
            "page"
        ]

        for anchor in page_result[
            "anchors"
        ]:

            item = anchor.copy()

            item["page"] = page

            all_anchors.append(
                item
            )

    # --------------------------------------------------------
    # Sort globally
    # --------------------------------------------------------

    all_anchors.sort(
        key=lambda a: (
            a["page"],
            a["bbox"][1]
        )
    )

    if not all_anchors:

        print(
            "\nNo question anchors detected."
        )

        return []

    # --------------------------------------------------------
    # Process every anchor
    # --------------------------------------------------------

    for i, current in enumerate(
        all_anchors
    ):

        qnum = current[
            "question_number"
        ]

        current_page = current[
            "page"
        ]

        current_page_result = (
            page_results[
                current_page - 1
            ]
        )

        current_image = cv2.imread(
            current_page_result[
                "image_path"
            ]
        )

        # ----------------------------------------------------
        # Find next question globally
        # ----------------------------------------------------

        next_anchor = None

        if i + 1 < len(all_anchors):

            next_anchor = all_anchors[
                i + 1
            ]

        # ----------------------------------------------------
        # CASE 1:
        # Next question is on same page
        # ----------------------------------------------------

        if (
            next_anchor is not None
            and
            next_anchor["page"]
            ==
            current_page
        ):

            crop = create_single_page_crop(
                current_image,
                current,
                next_anchor
            )

            filename = (
                f"Q{qnum}_"
                f"page_{current_page}.png"
            )

            path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            cv2.imwrite(
                path,
                crop
            )

            question_records.append({

                "question_number": qnum,

                "start_page": current_page,

                "end_page": current_page,

                "crop_path": path,

                "anchor_bbox":
                    current["bbox"],

                "anchor_text":
                    current["text"],

                "anchor_confidence":
                    current["confidence"],

                "type":
                    "single_page"

            })

            print(
                f"Created Q{qnum} "
                f"(page {current_page})"
            )

        # ----------------------------------------------------
        # CASE 2:
        # Question continues onto next page
        # ----------------------------------------------------

        else:

            # Current question begins on current page.

            h, w = current_image.shape[:2]

            x1, y1, x2, y2 = (
                current["bbox"]
            )

            crop_top = max(
                0,
                y1 - PAD_Y
            )

            # If no next question exists at all,
            # simply crop to bottom of current page.
            if next_anchor is None:

                crop = current_image[
                    crop_top:h,
                    0:w
                ]

                filename = (
                    f"Q{qnum}_"
                    f"page_{current_page}.png"
                )

                path = os.path.join(
                    OUTPUT_DIR,
                    filename
                )

                cv2.imwrite(
                    path,
                    crop
                )

                question_records.append({

                    "question_number":
                        qnum,

                    "start_page":
                        current_page,

                    "end_page":
                        current_page,

                    "crop_path":
                        path,

                    "anchor_bbox":
                        current["bbox"],

                    "anchor_text":
                        current["text"],

                    "anchor_confidence":
                        current[
                            "confidence"
                        ],

                    "type":
                        "final_question"

                })

                print(
                    f"Created Q{qnum} "
                    f"(final question)"
                )

            else:

                # ------------------------------------------------
                # For now:
                # save the starting-page portion.
                #
                # The continuation-page merging will be
                # handled separately after Student_1
                # detection is validated.
                # ------------------------------------------------

                crop = current_image[
                    crop_top:h,
                    0:w
                ]

                filename = (
                    f"Q{qnum}_"
                    f"page_{current_page}_"
                    f"continued.png"
                )

                path = os.path.join(
                    OUTPUT_DIR,
                    filename
                )

                cv2.imwrite(
                    path,
                    crop
                )

                question_records.append({

                    "question_number":
                        qnum,

                    "start_page":
                        current_page,

                    "end_page":
                        next_anchor["page"],

                    "crop_path":
                        path,

                    "anchor_bbox":
                        current["bbox"],

                    "anchor_text":
                        current["text"],

                    "anchor_confidence":
                        current[
                            "confidence"
                        ],

                    "type":
                        "possible_multi_page"

                })

                print(
                    f"Created Q{qnum} "
                    f"(possible multi-page)"
                )

    return question_records


# ============================================================
# NATURAL SORT
# ============================================================

def page_sort_key(filename):

    numbers = re.findall(
        r"\d+",
        filename
    )

    if numbers:

        return int(
            numbers[-1]
        )

    return 9999


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "ROBUST QUESTION DETECTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not os.path.exists(
        PAGE_DIR
    ):

        print(
            "\nERROR:"
        )

        print(
            "Page directory does not exist:"
        )

        print(
            PAGE_DIR
        )

        return

    # --------------------------------------------------------
    # Get page images
    # --------------------------------------------------------

    page_files = [

        filename

        for filename
        in os.listdir(PAGE_DIR)

        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        )
    ]

    page_files.sort(
        key=page_sort_key
    )

    if not page_files:

        print(
            "\nNo page images found."
        )

        return

    print(
        f"\nFound "
        f"{len(page_files)} pages."
    )

    for filename in page_files:

        print(
            "   ",
            filename
        )

    # --------------------------------------------------------
    # Process every page
    # --------------------------------------------------------

    page_results = []

    for index, filename in enumerate(
        page_files,
        start=1
    ):

        page_path = os.path.join(
            PAGE_DIR,
            filename
        )

        result = process_page(
            page_path,
            index
        )

        page_results.append(
            result
        )

    # --------------------------------------------------------
    # Create question crops
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "CREATING QUESTION CROPS"
    )
    print("=" * 70)

    question_records = (
        create_question_crops(
            page_results
        )
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "question_detection_results.json"
    )

    output = {

        "student_id":
            STUDENT_ID,

        "total_pages":
            len(page_results),

        "total_detected_questions":
            len(question_records),

        "questions":
            question_records,

        "pages":
            page_results
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 70)

    detected_numbers = [

        record[
            "question_number"
        ]

        for record
        in question_records
    ]

    print(
        "Detected question numbers:"
    )

    print(
        detected_numbers
    )

    print(
        f"\nTotal detected question regions:"
        f" {len(question_records)}"
    )

    # --------------------------------------------------------
    # Missing questions
    # --------------------------------------------------------

    expected = set(
        range(
            MIN_QUESTION,
            MAX_QUESTION + 1
        )
    )

    detected = set(
        detected_numbers
    )

    missing = sorted(
        expected - detected
    )

    duplicates = [

        q

        for q in detected_numbers

        if detected_numbers.count(q) > 1
    ]

    duplicates = sorted(
        set(duplicates)
    )

    print(
        "\nMissing question numbers:"
    )

    if missing:

        print(
            missing
        )

    else:

        print(
            "None"
        )

    print(
        "\nDuplicate question numbers:"
    )

    if duplicates:

        print(
            duplicates
        )

    else:

        print(
            "None"
        )

    print(
        "\nResults JSON:"
    )

    print(
        json_path
    )

    print(
        "\nQuestion crops:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nDebug images:"
    )

    print(
        DEBUG_DIR
    )

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()