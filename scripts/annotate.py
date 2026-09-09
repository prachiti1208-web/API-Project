import os
import json
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = "Student_6"

IMAGE_DIR = os.path.join(
    "exam_ocr_project",
    "dataset",
    "page_images",
    STUDENT_ID
)

OUTPUT_DIR = os.path.join(
    "exam_ocr_project",
    "annotations"
)
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{STUDENT_ID}_annotations.json"
)

MIN_QUESTION = 1
MAX_QUESTION = 35

MCQ_START = 1
MCQ_END = 20

SHORT_START = 21
SHORT_END = 35


# Maximum display size.
# Original image coordinates are preserved.
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850


# ============================================================
# GLOBAL STATE
# ============================================================

annotations = {}

current_page_index = 0
current_image = None
display_image = None

scale_x = 1.0
scale_y = 1.0

drawing = False
start_x = 0
start_y = 0

temp_box = None


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

def load_annotations():

    global annotations

    if os.path.exists(OUTPUT_FILE):

        try:

            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            print(f"\nLoaded existing annotations:")
            print(f"  {OUTPUT_FILE}")

        except Exception as e:

            print(f"\nCould not load existing annotations.")
            print(f"Error: {e}")

            annotations = {}

    else:

        annotations = {}


# ============================================================
# SAVE ANNOTATIONS
# ============================================================

def save_annotations():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            annotations,
            f,
            indent=4
        )

    print(f"\nAnnotations saved:")
    print(f"  {OUTPUT_FILE}")


# ============================================================
# FIND PAGE IMAGES
# ============================================================

def get_page_files():

    if not os.path.exists(IMAGE_DIR):

        print("\nERROR:")
        print(f"Image directory does not exist:")
        print(f"  {IMAGE_DIR}")
        return []

    valid_extensions = (
        ".png",
        ".jpg",
        ".jpeg"
    )

    files = []

    for filename in os.listdir(IMAGE_DIR):

        if filename.lower().endswith(valid_extensions):

            files.append(filename)

    files.sort()

    return files


# ============================================================
# GET QUESTION TYPE
# ============================================================

def get_question_type(question_number):

    if MCQ_START <= question_number <= MCQ_END:

        return "MCQ"

    if SHORT_START <= question_number <= SHORT_END:

        return "SHORT_ANSWER"

    return "UNKNOWN"


# ============================================================
# DISPLAY IMAGE
# ============================================================

def prepare_display_image(image):

    global scale_x
    global scale_y

    original_height, original_width = image.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / original_width,
        MAX_DISPLAY_HEIGHT / original_height,
        1.0
    )

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    display = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    scale_x = original_width / display_width
    scale_y = original_height / display_height

    return display


# ============================================================
# CONVERT DISPLAY COORDINATES
# TO ORIGINAL IMAGE COORDINATES
# ============================================================

def display_to_original(x, y):

    original_x = int(x * scale_x)
    original_y = int(y * scale_y)

    return original_x, original_y


# ============================================================
# DRAW SAVED ANNOTATIONS
# ============================================================

def draw_saved_annotations(image, page_name):

    page_annotations = annotations.get(
        page_name,
        []
    )

    for item in page_annotations:

        bbox = item.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Convert original coordinates
        # back to display coordinates.

        dx1 = int(x1 / scale_x)
        dy1 = int(y1 / scale_y)

        dx2 = int(x2 / scale_x)
        dy2 = int(y2 / scale_y)

        question_number = item.get(
            "question_number",
            "?"
        )

        q_type = item.get(
            "type",
            ""
        )

        cv2.rectangle(
            image,
            (dx1, dy1),
            (dx2, dy2),
            (0, 255, 0),
            2
        )

        label = f"Q{question_number} [{q_type}]"

        cv2.putText(
            image,
            label,
            (dx1, max(20, dy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global temp_box
    global display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        temp_box = None

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            temp_image = display_image.copy()

            cv2.rectangle(
                temp_image,
                (start_x, start_y),
                (x, y),
                (255, 0, 0),
                2
            )

            cv2.imshow(
                "Question Answer Annotation",
                temp_image
            )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        end_x = x
        end_y = y

        if abs(end_x - start_x) < 5:
            return

        if abs(end_y - start_y) < 5:
            return

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)

        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        temp_box = (
            x1,
            y1,
            x2,
            y2
        )


# ============================================================
# ADD ANNOTATION
# ============================================================

def add_annotation(
    page_name,
    display_box
):

    global annotations

    if display_box is None:

        print("\nNo box drawn.")

        return

    dx1, dy1, dx2, dy2 = display_box

    # Convert to original image coordinates.

    x1, y1 = display_to_original(
        dx1,
        dy1
    )

    x2, y2 = display_to_original(
        dx2,
        dy2
    )

    print("\n" + "-" * 60)

    while True:

        user_input = input(
            "Enter question number (1-35), or 'c' to cancel: "
        ).strip()

        if user_input.lower() == "c":

            print("Annotation cancelled.")

            return

        try:

            question_number = int(user_input)

        except ValueError:

            print("Please enter a number from 1 to 35.")

            continue

        if not (
            MIN_QUESTION
            <= question_number
            <= MAX_QUESTION
        ):

            print(
                f"Question number must be "
                f"{MIN_QUESTION}-{MAX_QUESTION}."
            )

            continue

        break

    q_type = get_question_type(
        question_number
    )

    annotation = {

        "question_number": question_number,

        "type": q_type,

        "bbox": [
            x1,
            y1,
            x2,
            y2
        ]
    }

    if page_name not in annotations:

        annotations[page_name] = []

    annotations[page_name].append(
        annotation
    )

    print(
        f"Added Q{question_number} "
        f"({q_type})"
    )

    print(
        f"Original bbox: "
        f"[{x1}, {y1}, {x2}, {y2}]"
    )

    save_annotations()


# ============================================================
# REDRAW PAGE
# ============================================================

def redraw_page(page_name):

    global display_image

    display_image = prepare_display_image(
        current_image
    )

    draw_saved_annotations(
        display_image,
        page_name
    )

    return display_image


# ============================================================
# UNDO LAST ANNOTATION
# ============================================================

def undo_last(page_name):

    if page_name not in annotations:

        print("\nNothing to undo.")

        return

    if len(annotations[page_name]) == 0:

        print("\nNothing to undo.")

        return

    removed = annotations[page_name].pop()

    print(
        f"\nRemoved Q"
        f"{removed['question_number']}"
    )

    if len(annotations[page_name]) == 0:

        del annotations[page_name]

    save_annotations()


# ============================================================
# RESET CURRENT PAGE
# ============================================================

def reset_page(page_name):

    if page_name not in annotations:

        print("\nNo annotations on this page.")

        return

    count = len(
        annotations[page_name]
    )

    answer = input(
        f"\nDelete {count} annotations "
        f"from {page_name}? (y/n): "
    ).strip().lower()

    if answer == "y":

        del annotations[page_name]

        save_annotations()

        print("Page annotations cleared.")

    else:

        print("Reset cancelled.")


# ============================================================
# PRINT PAGE STATUS
# ============================================================

def print_status(
    page_name,
    page_number,
    total_pages
):

    print("\n" + "=" * 70)

    print(
        f"Page {page_number + 1} "
        f"/ {total_pages}: "
        f"{page_name}"
    )

    print("=" * 70)

    page_annotations = annotations.get(
        page_name,
        []
    )

    if not page_annotations:

        print("Annotations: 0")

    else:

        print(
            f"Annotations: "
            f"{len(page_annotations)}"
        )

        for item in page_annotations:

            print(
                f"  Q{item['question_number']} "
                f"({item['type']})"
            )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    global current_page_index
    global current_image
    global display_image
    global temp_box

    print("\n" + "=" * 70)
    print("ANSWER REGION ANNOTATION TOOL")
    print("=" * 70)

    print("\nAnnotation rules:")

    print("\nQ1-Q20  → MCQ")

    print(
        "Draw a box around the complete "
        "student answer line."
    )

    print(
        "Example: [1. A]"
    )

    print("\nQ21-Q35 → SHORT ANSWER")

    print(
        "Draw a box around the ENTIRE "
        "handwritten answer."
    )

    print(
        "Do NOT box the question text "
        "from Question.txt."
    )

    print("\nKeyboard:")

    print("  N = next page")
    print("  P = previous page")
    print("  U = undo last annotation")
    print("  S = save")
    print("  R = reset current page")
    print("  Q = save and quit")

    print("=" * 70)

    # --------------------------------------------------------
    # Find pages
    # --------------------------------------------------------

    page_files = get_page_files()

    if not page_files:

        print("\nNo page images found.")

        return

    print("\nPages found:")

    for page in page_files:

        print(f"  {page}")

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    load_annotations()

    # --------------------------------------------------------
    # Open window
    # --------------------------------------------------------

    window_name = "Question Answer Annotation"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    # --------------------------------------------------------
    # Page loop
    # --------------------------------------------------------

    while True:

        page_name = page_files[
            current_page_index
        ]

        image_path = os.path.join(
            IMAGE_DIR,
            page_name
        )

        current_image = cv2.imread(
            image_path
        )

        if current_image is None:

            print(
                f"\nCould not read:"
                f" {image_path}"
            )

            current_page_index += 1

            if current_page_index >= len(page_files):

                break

            continue

        temp_box = None

        print_status(
            page_name,
            current_page_index,
            len(page_files)
        )

        display_image = redraw_page(
            page_name
        )

        cv2.imshow(
            window_name,
            display_image
        )

        print(
            "\nDraw a box with LEFT mouse button."
        )

        print(
            "Then enter the question number "
            "in the terminal."
        )

        print(
            "Press N/P/U/S/R/Q in the image window."
        )

        while True:

            key = cv2.waitKey(50) & 0xFF

            # ------------------------------------------------
            # No key
            # ------------------------------------------------

            if key == 255:
                continue

            # ------------------------------------------------
            # NEXT
            # ------------------------------------------------

            if key in (
                ord("n"),
                ord("N")
            ):

                if current_page_index < len(page_files) - 1:

                    current_page_index += 1

                    break

                else:

                    print(
                        "\nAlready on last page."
                    )

            # ------------------------------------------------
            # PREVIOUS
            # ------------------------------------------------

            elif key in (
                ord("p"),
                ord("P")
            ):

                if current_page_index > 0:

                    current_page_index -= 1

                    break

                else:

                    print(
                        "\nAlready on first page."
                    )

            # ------------------------------------------------
            # UNDO
            # ------------------------------------------------

            elif key in (
                ord("u"),
                ord("U")
            ):

                undo_last(page_name)

                display_image = redraw_page(
                    page_name
                )

                cv2.imshow(
                    window_name,
                    display_image
                )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            elif key in (
                ord("s"),
                ord("S")
            ):

                save_annotations()

            # ------------------------------------------------
            # RESET
            # ------------------------------------------------

            elif key in (
                ord("r"),
                ord("R")
            ):

                reset_page(page_name)

                display_image = redraw_page(
                    page_name
                )

                cv2.imshow(
                    window_name,
                    display_image
                )

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            elif key in (
                ord("q"),
                ord("Q")
            ):

                save_annotations()

                cv2.destroyAllWindows()

                print("\nAnnotation complete.")

                return

            # ------------------------------------------------
            # NEW ANNOTATION
            # ------------------------------------------------

            elif temp_box is not None:

                box = temp_box

                temp_box = None

                add_annotation(
                    page_name,
                    box
                )

                display_image = redraw_page(
                    page_name
                )

                cv2.imshow(
                    window_name,
                    display_image
                )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    save_annotations()

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("ANNOTATION FINISHED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()