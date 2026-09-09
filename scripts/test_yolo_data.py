
"""
DIAGNOSTIC YOLO VALIDATION TEST

Tests the trained YOLO model on:
    Student_1
    Student_6

Purpose:
    Find out exactly why some detected answers were skipped.

IMPORTANT:
    - Does NOT retrain YOLO.
    - Does NOT refine crop boundaries.
    - Keeps the original YOLO detections.
    - Separately records size-filter removals.
    - Separately records NMS removals.
    - Saves diagnostic images and crops.

Pipeline:

    YOLO
      |
      +----> ALL RAW DETECTIONS
      |
      +----> SIZE FILTER
      |          |
      |          +----> REMOVED BY SIZE
      |          |
      |          +----> PASSED SIZE
      |
      +----> NMS
                 |
                 +----> REMOVED BY NMS
                 |
                 +----> FINAL DETECTIONS
"""

import os
import json
import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "exam_ocr_project",
    "runs",
    "answer_detection",
    "yolo_answer_detector",
    "weights",
    "best.pt"
)

PAGE_IMAGE_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "page_images"
)

OUTPUT_DIR = os.path.join(
    "exam_ocr_project",
    "runs",
    "answer_detection",
    "yolo_diagnostic_test"
)

VALIDATION_STUDENTS = [
    "Student_1",
    "Student_6"
]


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 1024


# ============================================================
# NMS SETTINGS
# ============================================================

# We keep NMS relatively conservative for diagnosis.
NMS_IOU_THRESHOLD = 0.40


# ============================================================
# SIZE FILTER SETTINGS
# ============================================================

MCQ_MIN_WIDTH = 40
MCQ_MIN_HEIGHT = 30
MCQ_MAX_WIDTH = 1200
MCQ_MAX_HEIGHT = 700

SHORT_MIN_WIDTH = 150
SHORT_MIN_HEIGHT = 80
SHORT_MAX_WIDTH = 4700
SHORT_MAX_HEIGHT = 1800

MAX_PAGE_AREA_RATIO = 0.45


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# SIZE FILTER
# ============================================================

def get_size_filter_reason(
    box,
    class_name,
    page_width,
    page_height
):

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    box_area = box_width * box_height

    page_area = page_width * page_height

    area_ratio = (
        box_area / page_area
        if page_area > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Page area
    # --------------------------------------------------------

    if area_ratio > MAX_PAGE_AREA_RATIO:
        return (
            f"PAGE_AREA_RATIO_TOO_LARGE "
            f"({area_ratio:.3f})"
        )

    # --------------------------------------------------------
    # MCQ
    # --------------------------------------------------------

    if class_name == "MCQ":

        if box_width < MCQ_MIN_WIDTH:
            return f"MCQ_WIDTH_TOO_SMALL ({box_width})"

        if box_height < MCQ_MIN_HEIGHT:
            return f"MCQ_HEIGHT_TOO_SMALL ({box_height})"

        if box_width > MCQ_MAX_WIDTH:
            return f"MCQ_WIDTH_TOO_LARGE ({box_width})"

        if box_height > MCQ_MAX_HEIGHT:
            return f"MCQ_HEIGHT_TOO_LARGE ({box_height})"

    # --------------------------------------------------------
    # SHORT ANSWER
    # --------------------------------------------------------

    elif class_name == "SHORT_ANSWER":

        if box_width < SHORT_MIN_WIDTH:
            return (
                f"SHORT_WIDTH_TOO_SMALL "
                f"({box_width})"
            )

        if box_height < SHORT_MIN_HEIGHT:
            return (
                f"SHORT_HEIGHT_TOO_SMALL "
                f"({box_height})"
            )

        if box_width > SHORT_MAX_WIDTH:
            return (
                f"SHORT_WIDTH_TOO_LARGE "
                f"({box_width})"
            )

        if box_height > SHORT_MAX_HEIGHT:
            return (
                f"SHORT_HEIGHT_TOO_LARGE "
                f"({box_height})"
            )

    return None


# ============================================================
# NMS
# ============================================================

def remove_duplicate_boxes(detections):

    cleaned = []
    removed = []

    class_names = set(
        detection["class"]
        for detection in detections
    )

    for class_name in class_names:

        class_detections = [
            detection
            for detection in detections
            if detection["class"] == class_name
        ]

        class_detections.sort(
            key=lambda d: d["confidence"],
            reverse=True
        )

        kept = []

        for candidate in class_detections:

            candidate_box = candidate["bbox"]

            duplicate_of = None
            duplicate_iou = 0.0

            for existing in kept:

                existing_box = existing["bbox"]

                iou = calculate_iou(
                    candidate_box,
                    existing_box
                )

                if iou >= NMS_IOU_THRESHOLD:

                    duplicate_of = (
                        existing["raw_index"]
                    )

                    duplicate_iou = iou

                    break

            if duplicate_of is not None:

                candidate_copy = dict(candidate)

                candidate_copy["removal_reason"] = (
                    "NMS_DUPLICATE"
                )

                candidate_copy["duplicate_of"] = (
                    duplicate_of
                )

                candidate_copy["iou"] = round(
                    duplicate_iou,
                    4
                )

                removed.append(candidate_copy)

            else:

                kept.append(candidate)

        cleaned.extend(kept)

    cleaned.sort(
        key=lambda d: (
            d["bbox"][1],
            d["bbox"][0]
        )
    )

    for index, detection in enumerate(
        cleaned,
        start=1
    ):
        detection["detection_index"] = index

    return cleaned, removed


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections,
    label_prefix=""
):

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        confidence = detection["confidence"]
        class_name = detection["class"]

        label = (
            f"{label_prefix}"
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Loading YOLO model...")
print("=" * 70)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:", model.names)


# ============================================================
# STORE RESULTS
# ============================================================

all_results = {}


# ============================================================
# PROCESS STUDENTS
# ============================================================

for student_id in VALIDATION_STUDENTS:

    print("\n" + "=" * 70)
    print(f"PROCESSING {student_id}")
    print("=" * 70)

    student_input_dir = os.path.join(
        PAGE_IMAGE_DIR,
        student_id
    )

    student_output_dir = os.path.join(
        OUTPUT_DIR,
        student_id
    )

    os.makedirs(
        student_output_dir,
        exist_ok=True
    )

    raw_crop_dir = os.path.join(
        student_output_dir,
        "01_raw_crops"
    )

    size_removed_crop_dir = os.path.join(
        student_output_dir,
        "02_size_removed"
    )

    final_crop_dir = os.path.join(
        student_output_dir,
        "03_final_crops"
    )

    os.makedirs(
        raw_crop_dir,
        exist_ok=True
    )

    os.makedirs(
        size_removed_crop_dir,
        exist_ok=True
    )

    os.makedirs(
        final_crop_dir,
        exist_ok=True
    )

    if not os.path.exists(student_input_dir):

        print(
            f"WARNING: Folder not found: "
            f"{student_input_dir}"
        )

        continue

    page_files = [
        file
        for file in os.listdir(student_input_dir)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    page_files.sort()

    student_results = {}

    # ========================================================
    # PROCESS EACH PAGE
    # ========================================================

    for page_file in page_files:

        print("\n" + "-" * 70)
        print(f"Processing: {page_file}")
        print("-" * 70)

        image_path = os.path.join(
            student_input_dir,
            page_file
        )

        image = cv2.imread(image_path)

        if image is None:

            print(
                "WARNING: Could not read image."
            )

            continue

        original_image = image.copy()

        page_height, page_width = (
            original_image.shape[:2]
        )

        # ====================================================
        # YOLO PREDICTION
        # ====================================================

        results = model.predict(
            source=original_image,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        # ====================================================
        # ALL RAW YOLO DETECTIONS
        # ====================================================

        all_raw_detections = []

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for raw_index, (
                box,
                confidence,
                class_id
            ) in enumerate(
                zip(
                    boxes,
                    confidences,
                    class_ids
                ),
                start=1
            ):

                x1, y1, x2, y2 = (
                    box.astype(int)
                )

                confidence = float(
                    confidence
                )

                class_id = int(
                    class_id
                )

                class_name = model.names[
                    class_id
                ]

                # Clamp coordinates
                x1 = max(
                    0,
                    min(
                        x1,
                        page_width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        page_height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        page_width
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        page_height
                    )
                )

                if x2 <= x1:
                    continue

                if y2 <= y1:
                    continue

                detection = {
                    "raw_index": raw_index,
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                }

                all_raw_detections.append(
                    detection
                )

        # ====================================================
        # PRINT TRUE RAW COUNT
        # ====================================================

        print(
            f"YOLO RAW DETECTIONS: "
            f"{len(all_raw_detections)}"
        )

        # ====================================================
        # SAVE ALL RAW CROPS
        # ====================================================

        page_stem = os.path.splitext(
            page_file
        )[0]

        for detection in all_raw_detections:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            crop = original_image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            raw_filename = (
                f"{page_stem}"
                f"_raw_"
                f"{detection['raw_index']:03d}_"
                f"{detection['class']}.png"
            )

            raw_path = os.path.join(
                raw_crop_dir,
                raw_filename
            )

            cv2.imwrite(
                raw_path,
                crop
            )

        # ====================================================
        # SIZE FILTER
        # ====================================================

        size_passed = []
        size_removed = []

        for detection in all_raw_detections:

            reason = get_size_filter_reason(
                detection["bbox"],
                detection["class"],
                page_width,
                page_height
            )

            if reason is not None:

                removed_detection = dict(
                    detection
                )

                removed_detection[
                    "removal_reason"
                ] = reason

                size_removed.append(
                    removed_detection
                )

            else:

                size_passed.append(
                    detection
                )

        # ====================================================
        # PRINT SIZE FILTER RESULT
        # ====================================================

        print(
            f"SIZE FILTER REMOVED: "
            f"{len(size_removed)}"
        )

        print(
            f"PASSED SIZE FILTER: "
            f"{len(size_passed)}"
        )

        # ====================================================
        # DETAILS OF SIZE REMOVALS
        # ====================================================

        for detection in size_removed:

            print(
                "   SIZE REMOVED:"
                f" raw={detection['raw_index']}"
                f" class={detection['class']}"
                f" conf={detection['confidence']:.2f}"
                f" bbox={detection['bbox']}"
                f" reason={detection['removal_reason']}"
            )

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            crop = original_image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            filename = (
                f"{page_stem}"
                f"_SIZE_REMOVED_"
                f"{detection['raw_index']:03d}_"
                f"{detection['class']}.png"
            )

            path = os.path.join(
                size_removed_crop_dir,
                filename
            )

            cv2.imwrite(
                path,
                crop
            )

        # ====================================================
        # NMS
        # ====================================================

        final_detections, nms_removed = (
            remove_duplicate_boxes(
                size_passed
            )
        )

        print(
            f"NMS REMOVED: "
            f"{len(nms_removed)}"
        )

        print(
            f"FINAL DETECTIONS: "
            f"{len(final_detections)}"
        )

        # ====================================================
        # DETAILS OF NMS REMOVALS
        # ====================================================

        for detection in nms_removed:

            print(
                "   NMS REMOVED:"
                f" raw={detection['raw_index']}"
                f" class={detection['class']}"
                f" conf={detection['confidence']:.2f}"
                f" bbox={detection['bbox']}"
                f" duplicate_of="
                f"{detection['duplicate_of']}"
                f" IoU={detection['iou']:.3f}"
            )

        # ====================================================
        # SAVE FINAL CROPS
        # ====================================================

        for detection_index, detection in enumerate(
            final_detections,
            start=1
        ):

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            crop = original_image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            filename = (
                f"{page_stem}"
                f"_final_"
                f"{detection_index:03d}_"
                f"{detection['class']}.png"
            )

            crop_path = os.path.join(
                final_crop_dir,
                filename
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            detection[
                "crop_path"
            ] = os.path.abspath(
                crop_path
            )

        # ====================================================
        # DRAW ALL RAW DETECTIONS
        # ====================================================

        raw_visual = draw_detections(
            original_image,
            all_raw_detections,
            label_prefix="RAW "
        )

        raw_visual_path = os.path.join(
            student_output_dir,
            f"raw_{page_file}"
        )

        cv2.imwrite(
            raw_visual_path,
            raw_visual
        )

        # ====================================================
        # DRAW FINAL DETECTIONS
        # ====================================================

        final_visual = draw_detections(
            original_image,
            final_detections,
            label_prefix="FINAL "
        )

        final_visual_path = os.path.join(
            student_output_dir,
            f"final_{page_file}"
        )

        cv2.imwrite(
            final_visual_path,
            final_visual
        )

        # ====================================================
        # COUNTS
        # ====================================================

        raw_mcq = sum(
            1
            for d in all_raw_detections
            if d["class"] == "MCQ"
        )

        raw_short = sum(
            1
            for d in all_raw_detections
            if d["class"] == "SHORT_ANSWER"
        )

        final_mcq = sum(
            1
            for d in final_detections
            if d["class"] == "MCQ"
        )

        final_short = sum(
            1
            for d in final_detections
            if d["class"] == "SHORT_ANSWER"
        )

        # ====================================================
        # PAGE RESULTS
        # ====================================================

        student_results[page_file] = {

            "page_width": page_width,
            "page_height": page_height,

            "raw_count": len(
                all_raw_detections
            ),

            "size_removed_count": len(
                size_removed
            ),

            "size_passed_count": len(
                size_passed
            ),

            "nms_removed_count": len(
                nms_removed
            ),

            "final_count": len(
                final_detections
            ),

            "raw_detections":
                all_raw_detections,

            "size_removed":
                size_removed,

            "nms_removed":
                nms_removed,

            "final_detections":
                final_detections,

            "raw_mcq": raw_mcq,
            "raw_short_answer": raw_short,

            "final_mcq": final_mcq,
            "final_short_answer": final_short
        }

        # ====================================================
        # PAGE SUMMARY
        # ====================================================

        print()
        print(
            "SUMMARY:"
        )
        print(
            f"   Raw YOLO:        "
            f"{len(all_raw_detections)}"
        )
        print(
            f"   Size removed:    "
            f"{len(size_removed)}"
        )
        print(
            f"   Size passed:     "
            f"{len(size_passed)}"
        )
        print(
            f"   NMS removed:     "
            f"{len(nms_removed)}"
        )
        print(
            f"   Final:           "
            f"{len(final_detections)}"
        )

        print(
            f"   Raw MCQ:         {raw_mcq}"
        )
        print(
            f"   Raw SHORT:       {raw_short}"
        )
        print(
            f"   Final MCQ:       {final_mcq}"
        )
        print(
            f"   Final SHORT:     {final_short}"
        )

    all_results[student_id] = (
        student_results
    )


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "diagnostic_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC YOLO TEST COMPLETE")
print("=" * 70)

for student_id, pages in all_results.items():

    raw_total = sum(
        page["raw_count"]
        for page in pages.values()
    )

    size_removed_total = sum(
        page["size_removed_count"]
        for page in pages.values()
    )

    nms_removed_total = sum(
        page["nms_removed_count"]
        for page in pages.values()
    )

    final_total = sum(
        page["final_count"]
        for page in pages.values()
    )

    print()
    print(student_id)

    print(
        f"   YOLO RAW:       "
        f"{raw_total}"
    )

    print(
        f"   SIZE REMOVED:   "
        f"{size_removed_total}"
    )

    print(
        f"   NMS REMOVED:    "
        f"{nms_removed_total}"
    )

    print(
        f"   FINAL:          "
        f"{final_total}"
    )

print()
print("=" * 70)

print("Output folder:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print()
print("Diagnostic JSON:")
print(
    os.path.abspath(json_path)
)

print()
print("Important files to inspect:")

print(
    "1. raw_page_0.png"
)

print(
    "2. final_page_0.png"
)

print(
    "3. 02_size_removed/"
)

print(
    "4. 03_final_crops/"
)

print()
print(
    "DO NOT run the short-answer refinement yet."
)

print(
    "First inspect which detections were removed."
)

