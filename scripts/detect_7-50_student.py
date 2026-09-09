from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

# Trained YOLO model
MODEL_PATH = Path(
    r"C:\Users\jpras\OneDrive\Documents\exam_ocr_project\exam_ocr_project\runs\answer_detection\yolo_answer_detector\weights\best.pt"
)

# Folder containing Student_1 ... Student_50
INPUT_ROOT = Path(
    r"C:\Users\jpras\OneDrive\Documents\exam_ocr_project\exam_ocr_project\dataset\page_images"
)

# Output folder
OUTPUT_ROOT = Path(
    r"C:\Users\jpras\OneDrive\Documents\exam_ocr_project\exam_ocr_project\runs\answer_detection\students_7_to_50"
)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Students to process
START_STUDENT = 7
END_STUDENT = 50


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING YOLO MODEL")
print("=" * 70)

model = YOLO(str(MODEL_PATH))

print(f"Model loaded: {MODEL_PATH}")


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS STUDENTS 7 TO 50
# ============================================================

for student_number in range(
    START_STUDENT,
    END_STUDENT + 1
):

    student_name = f"Student_{student_number}"

    student_dir = INPUT_ROOT / student_name

    print("\n" + "=" * 70)
    print(f"PROCESSING: {student_name}")
    print("=" * 70)


    # --------------------------------------------------------
    # CHECK STUDENT FOLDER
    # --------------------------------------------------------

    if not student_dir.exists():

        print(
            f"[WARNING] Folder not found: {student_dir}"
        )

        continue


    # --------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # --------------------------------------------------------

    student_output_dir = (
        OUTPUT_ROOT / student_name
    )

    crops_dir = (
        student_output_dir / "crops"
    )

    crops_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # FIND PAGE IMAGES
    # --------------------------------------------------------

    image_paths = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg"
    ]:

        image_paths.extend(
            student_dir.glob(extension)
        )


    # Sort pages
    image_paths = sorted(image_paths)


    print(
        f"Pages found: {len(image_paths)}"
    )


    # --------------------------------------------------------
    # PROCESS EACH PAGE
    # --------------------------------------------------------

    for page_path in image_paths:


        print(
            f"\nProcessing page: {page_path.name}"
        )


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(page_path)
        )


        if image is None:

            print(
                f"[WARNING] Cannot read image: {page_path}"
            )

            continue


        # ----------------------------------------------------
        # YOLO PREDICTION
        # ----------------------------------------------------

        results = model.predict(

            source=str(page_path),

            conf=CONFIDENCE_THRESHOLD,

            verbose=False
        )


        result = results[0]


        # ----------------------------------------------------
        # GET DETECTIONS
        # ----------------------------------------------------

        detections = []


        if result.boxes is not None:


            for box in result.boxes:


                # Bounding box
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                # Confidence
                confidence = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )


                # Class ID
                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )


                # Class name
                class_name = (
                    model.names[class_id]
                )


                detections.append({

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "confidence": confidence,

                    "class_id": class_id,

                    "class_name": class_name

                })


        # ----------------------------------------------------
        # SORT DETECTIONS TOP TO BOTTOM
        # ----------------------------------------------------

        detections = sorted(

            detections,

            key=lambda d: d["y1"]
        )


        # ----------------------------------------------------
        # CREATE CROPS
        # ----------------------------------------------------

        for index, detection in enumerate(
            detections,
            start=1
        ):


            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]


            class_name = (
                detection["class_name"]
            )


            confidence = (
                detection["confidence"]
            )


            # ----------------------------------------------
            # SAFETY BOUNDARIES
            # ----------------------------------------------

            height, width = image.shape[:2]


            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)


            # ----------------------------------------------
            # CROP
            # ----------------------------------------------

            crop = image[
                y1:y2,
                x1:x2
            ]


            if crop.size == 0:

                print(
                    f"[WARNING] Empty crop skipped: "
                    f"{page_path.name}"
                )

                continue


            # ----------------------------------------------
            # SAVE CROP
            # ----------------------------------------------

            crop_filename = (

                f"{page_path.stem}"
                f"_answer_{index:03d}"
                f"_{class_name}.png"

            )


            crop_path = (

                crops_dir /
                crop_filename

            )


            cv2.imwrite(

                str(crop_path),

                crop

            )


            print(

                f"  Saved: "
                f"{crop_filename} "
                f"| {class_name} "
                f"| {confidence:.3f}"

            )


        # ----------------------------------------------------
        # SAVE DETECTED PAGE
        # ----------------------------------------------------

        annotated_image = (
            result.plot()
        )


        annotated_page_path = (

            student_output_dir /
            f"{page_path.stem}_detected.png"

        )


        cv2.imwrite(

            str(annotated_page_path),

            annotated_image

        )


        print(
            f"Detections: {len(detections)}"
        )


    print(
        f"\nCompleted: {student_name}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("ALL STUDENTS PROCESSED")
print("=" * 70)

print(
    f"Students: "
    f"{START_STUDENT} to {END_STUDENT}"
)

print(
    f"Output: {OUTPUT_ROOT}"
)