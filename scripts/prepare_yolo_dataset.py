import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
PAGE_IMAGES_DIR = PROJECT_ROOT / "dataset" / "page_images"

YOLO_DATASET_DIR = PROJECT_ROOT / "dataset" / "yolo_dataset"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAP = {
    "MCQ": 0,
    "SHORT_ANSWER": 1
}


# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

def clean_output_directory():

    if YOLO_DATASET_DIR.exists():

        print("\nCleaning old YOLO dataset...")

        shutil.rmtree(YOLO_DATASET_DIR)

    # Create directory structure

    directories = [

        YOLO_DATASET_DIR / "images" / "train",
        YOLO_DATASET_DIR / "images" / "val",

        YOLO_DATASET_DIR / "labels" / "train",
        YOLO_DATASET_DIR / "labels" / "val"
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# FIND ANNOTATION FILES
# ============================================================

def get_annotation_files():

    annotation_files = sorted(
        ANNOTATIONS_DIR.glob(
            "Student_*_annotations.json"
        )
    )

    return annotation_files


# ============================================================
# READ ONE STUDENT'S ANNOTATIONS
# ============================================================

def load_annotations(annotation_file):

    with open(
        annotation_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


# ============================================================
# CONVERT BBOX TO YOLO FORMAT
# ============================================================

def convert_bbox_to_yolo(
    bbox,
    image_width,
    image_height
):

    x1, y1, x2, y2 = bbox


    # --------------------------------------------------------
    # Ensure correct coordinate ordering
    # --------------------------------------------------------

    x1 = min(max(x1, 0), image_width)
    x2 = min(max(x2, 0), image_width)

    y1 = min(max(y1, 0), image_height)
    y2 = min(max(y2, 0), image_height)


    # --------------------------------------------------------
    # Calculate width and height
    # --------------------------------------------------------

    bbox_width = x2 - x1
    bbox_height = y2 - y1


    # Invalid bounding box

    if bbox_width <= 0 or bbox_height <= 0:

        return None


    # --------------------------------------------------------
    # Calculate center
    # --------------------------------------------------------

    center_x = x1 + (bbox_width / 2)
    center_y = y1 + (bbox_height / 2)


    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    center_x /= image_width
    center_y /= image_height

    bbox_width /= image_width
    bbox_height /= image_height


    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    values = [

        center_x,
        center_y,
        bbox_width,
        bbox_height

    ]


    if not all(
        0 <= value <= 1
        for value in values
    ):

        return None


    return (

        center_x,
        center_y,
        bbox_width,
        bbox_height

    )


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    student_id,
    page_name,
    annotations,
    split
):

    image_path = (

        PAGE_IMAGES_DIR
        / student_id
        / page_name

    )


    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not image_path.exists():

        print(
            f"⚠ Image not found: {image_path}"
        )

        return 0


    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        with Image.open(image_path) as image:

            image_width, image_height = image.size

    except Exception as e:

        print(
            f"⚠ Could not open image: "
            f"{image_path}"
        )

        print(e)

        return 0


    # --------------------------------------------------------
    # Convert annotations
    # --------------------------------------------------------

    yolo_labels = []


    for annotation in annotations:


        annotation_type = annotation.get("type")


        # Skip unknown classes

        if annotation_type not in CLASS_MAP:

            print(

                f"⚠ Unknown type: "
                f"{annotation_type}"

            )

            continue


        bbox = annotation.get("bbox")


        if not bbox or len(bbox) != 4:

            print(

                f"⚠ Invalid bbox in "
                f"{student_id} | "
                f"{page_name}"

            )

            continue


        # ----------------------------------------------------
        # Convert bbox
        # ----------------------------------------------------

        yolo_bbox = convert_bbox_to_yolo(

            bbox,
            image_width,
            image_height

        )


        if yolo_bbox is None:

            print(

                f"⚠ Invalid converted bbox in "
                f"{student_id} | "
                f"{page_name}"

            )

            continue


        class_id = CLASS_MAP[annotation_type]


        center_x, center_y, width, height = yolo_bbox


        # ----------------------------------------------------
        # YOLO FORMAT
        #
        # class x_center y_center width height
        # ----------------------------------------------------

        label_line = (

            f"{class_id} "

            f"{center_x:.6f} "

            f"{center_y:.6f} "

            f"{width:.6f} "

            f"{height:.6f}"

        )


        yolo_labels.append(
            label_line
        )


    # --------------------------------------------------------
    # Skip pages with no annotations
    # --------------------------------------------------------

    if len(yolo_labels) == 0:

        print(

            f"⚠ No valid annotations: "
            f"{student_id} | "
            f"{page_name}"

        )

        return 0


    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    page_stem = Path(page_name).stem


    output_name = (

        f"{student_id}_"
        f"{page_stem}"

    )


    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_image_path = (

        YOLO_DATASET_DIR
        / "images"
        / split
        / f"{output_name}.png"

    )


    output_label_path = (

        YOLO_DATASET_DIR
        / "labels"
        / split
        / f"{output_name}.txt"

    )


    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(

        image_path,
        output_image_path

    )


    # --------------------------------------------------------
    # Save YOLO labels
    # --------------------------------------------------------

    with open(

        output_label_path,
        "w",
        encoding="utf-8"

    ) as f:


        for label in yolo_labels:

            f.write(
                label + "\n"
            )


    print(

        f"✓ {student_id} | "
        f"{page_name} | "
        f"{len(yolo_labels)} boxes | "
        f"{split}"

    )


    return len(yolo_labels)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PREPARING YOLO DATASET"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    clean_output_directory()


    # --------------------------------------------------------
    # Find annotation files
    # --------------------------------------------------------

    annotation_files = (
        get_annotation_files()
    )


    if len(annotation_files) == 0:

        print(
            "\n❌ No annotation files found."
        )

        print(
            f"Expected directory:\n"
            f"{ANNOTATIONS_DIR}"
        )

        return


    print(
        f"\nAnnotation files found: "
        f"{len(annotation_files)}"
    )


    # --------------------------------------------------------
    # Shuffle students
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)


    random.shuffle(
        annotation_files
    )


    # --------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------

    total_students = (
        len(annotation_files)
    )


    train_count = max(

        1,

        int(
            total_students
            * TRAIN_RATIO
        )

    )


    train_files = (
        annotation_files[:train_count]
    )


    val_files = (
        annotation_files[train_count:]
    )


    # Ensure validation has at least one student

    if len(val_files) == 0:

        val_files = [

            train_files.pop()

        ]


    print(
        "\nTRAIN STUDENTS:"
    )


    for file in train_files:

        print(
            f"  {file.stem}"
        )


    print(
        "\nVALIDATION STUDENTS:"
    )


    for file in val_files:

        print(
            f"  {file.stem}"
        )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_pages = 0
    total_boxes = 0


    # ========================================================
    # PROCESS TRAIN DATA
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSING TRAIN DATA"
    )

    print(
        "=" * 70
    )


    for annotation_file in train_files:


        student_id = (

            annotation_file
            .stem
            .replace(
                "_annotations",
                ""
            )

        )


        data = load_annotations(
            annotation_file
        )


        for page_name, annotations in data.items():


            boxes = process_page(

                student_id,
                page_name,
                annotations,
                "train"

            )


            if boxes > 0:

                total_pages += 1

                total_boxes += boxes


    # ========================================================
    # PROCESS VALIDATION DATA
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSING VALIDATION DATA"
    )

    print(
        "=" * 70
    )


    for annotation_file in val_files:


        student_id = (

            annotation_file
            .stem
            .replace(
                "_annotations",
                ""
            )

        )


        data = load_annotations(
            annotation_file
        )


        for page_name, annotations in data.items():


            boxes = process_page(

                student_id,
                page_name,
                annotations,
                "val"

            )


            if boxes > 0:

                total_pages += 1

                total_boxes += boxes


    # ========================================================
    # CREATE data.yaml
    # ========================================================

    yaml_path = (

        YOLO_DATASET_DIR
        / "data.yaml"

    )


    dataset_path = (
        YOLO_DATASET_DIR.resolve()
    )


    yaml_content = (

        f"path: {dataset_path.as_posix()}\n"

        f"train: images/train\n"

        f"val: images/val\n\n"

        f"names:\n"

        f"  0: MCQ\n"

        f"  1: SHORT_ANSWER\n"

    )


    with open(

        yaml_path,
        "w",
        encoding="utf-8"

    ) as f:

        f.write(
            yaml_content
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    train_images = list(

        (
            YOLO_DATASET_DIR
            / "images"
            / "train"

        ).glob("*")

    )


    val_images = list(

        (
            YOLO_DATASET_DIR
            / "images"
            / "val"

        ).glob("*")

    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal students: "
        f"{total_students}"
    )

    print(
        f"Training students: "
        f"{len(train_files)}"
    )

    print(
        f"Validation students: "
        f"{len(val_files)}"
    )

    print(
        f"\nTotal annotated pages: "
        f"{total_pages}"
    )

    print(
        f"Total bounding boxes: "
        f"{total_boxes}"
    )

    print(
        f"\nTraining images: "
        f"{len(train_images)}"
    )

    print(
        f"Validation images: "
        f"{len(val_images)}"
    )

    print(
        f"\nDataset saved in:"
    )

    print(
        YOLO_DATASET_DIR
    )

    print(
        f"\nData configuration:"
    )

    print(
        yaml_path
    )


if __name__ == "__main__":

    main()