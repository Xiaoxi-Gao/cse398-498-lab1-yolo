import csv
import os
import re
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO


# =====================================
# Experiment 1 settings
# =====================================

MODEL_PATH = "./models/yolo11n.pt"

# External camera
CAM_INDEX = 1

# YOLO confidence threshold
CONF = 0.50

OUT_DIR = "results/exp1_baseline"

RAW_DIR = os.path.join(OUT_DIR, "raw")
ANNOTATED_DIR = os.path.join(OUT_DIR, "annotated")

CSV_PATH = os.path.join(OUT_DIR, "detections.csv")


# =====================================
# Create folders
# =====================================

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


# =====================================
# Create CSV
# =====================================

if not os.path.exists(CSV_PATH):

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "timestamp",
            "ground_truth",
            "predicted_class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "conf_threshold",
            "evaluation",
            "notes"
        ])


# =====================================
# Find next test number
# =====================================

def get_next_test_number():

    used_numbers = []

    for filename in os.listdir(RAW_DIR):

        match = re.fullmatch(
            r"test(\d+)\.jpg",
            filename
        )

        if match:
            used_numbers.append(
                int(match.group(1))
            )

    if used_numbers:
        return max(used_numbers) + 1

    return 1


# =====================================
# Load pretrained YOLO11
# =====================================

model = YOLO(MODEL_PATH)


# =====================================
# Open external camera
# =====================================

cap = cv2.VideoCapture(
    CAM_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    sys.exit(
        "Could not open camera. "
        "Close Zoom/Teams or check camera index."
    )


test_number = get_next_test_number()

print(
    f"Ready. Next image: test{test_number:02d}"
)

print("Press S to save")
print("Press Q to quit")


# =====================================
# Main detection loop
# =====================================

while True:

    success, frame = cap.read()

    if not success:
        continue

    # Save clean frame before drawing boxes
    raw_frame = frame.copy()


    # Run pretrained YOLO11
    result = model.predict(
        frame,
        conf=CONF,
        verbose=False
    )[0]


    current_detections = []


    # =================================
    # Draw detections
    # =================================

    for box, score, cls in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        label = model.names[
            int(cls)
        ]

        confidence = float(score)


        current_detections.append([
            label,
            round(confidence, 4),
            x1,
            y1,
            x2,
            y2
        ])


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # =================================
    # Display information
    # =================================

    hud = (
        f"threshold={CONF:.2f} "
        f"detections={len(current_detections)} "
        f"next=test{test_number:02d}"
    )

    cv2.putText(
        frame,
        hud,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "YOLO11 Baseline - Experiment 1",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # =================================
    # Quit
    # =================================

    if key == ord("q"):
        break


    # =================================
    # Save one test image
    # =================================

    elif key == ord("s"):

        image_id = (
            f"test{test_number:02d}"
        )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )


        # Save raw image
        raw_path = os.path.join(
            RAW_DIR,
            image_id + ".jpg"
        )

        cv2.imwrite(
            raw_path,
            raw_frame
        )


        # Save annotated image
        annotated_path = os.path.join(
            ANNOTATED_DIR,
            image_id + ".jpg"
        )

        cv2.imwrite(
            annotated_path,
            frame
        )


        # =================================
        # Save detection results to CSV
        # =================================

        with open(
            CSV_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)


            if current_detections:

                for (
                    label,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                ) in current_detections:

                    writer.writerow([
                        image_id,
                        timestamp,
                        "",
                        label,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        CONF,
                        "",
                        ""
                    ])


            else:

                writer.writerow([
                    image_id,
                    timestamp,
                    "",
                    "NO_DETECTION",
                    "",
                    "",
                    "",
                    "",
                    "",
                    CONF,
                    "",
                    ""
                ])


        print(
            f"Saved {image_id}: "
            f"{len(current_detections)} detections"
        )


        test_number += 1


# =====================================
# Cleanup
# =====================================

cap.release()

cv2.destroyAllWindows()

print(
    "Experiment finished."
)

print(
    f"Results saved in: {OUT_DIR}"
)