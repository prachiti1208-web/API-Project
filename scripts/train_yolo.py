from ultralytics import YOLO
import os


def main():

    print("=" * 70)
    print("TRAINING YOLO FOR HANDWRITTEN ANSWER REGION DETECTION")
    print("=" * 70)

    # --------------------------------------------------
    # PROJECT PATHS
    # --------------------------------------------------

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    DATA_YAML = os.path.join(
        BASE_DIR,
        "dataset",
        "yolo_dataset",
        "data.yaml"
    )

    # --------------------------------------------------
    # LOAD YOLO MODEL
    # --------------------------------------------------

    print("\nLoading YOLOv8 model...")

    # Small model is good for initial testing
    model = YOLO("yolov8n.pt")

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    print("\nStarting training...\n")

    results = model.train(

        data=DATA_YAML,

        epochs=50,

        imgsz=1024,

        batch=2,

        device="cpu",

        workers=0,

        project=os.path.join(
            BASE_DIR,
            "runs",
            "answer_detection"
        ),

        name="yolo_answer_detector",

        patience=15,

        pretrained=True,

        verbose=True
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nResults saved in:")

    print(
        os.path.join(
            BASE_DIR,
            "runs",
            "answer_detection",
            "yolo_answer_detector"
        )
    )


if __name__ == "__main__":
    main()