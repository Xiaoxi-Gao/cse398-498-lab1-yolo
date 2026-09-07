import os
import csv
import cv2
from ultralytics import YOLO

# =========================
# 1. Path Configuration
# =========================

# Path to the latest fine-tuned model
MODEL_PATH = "../models/yolo11n_finetuned_final.pt"

# Raw test images saved from Experiment 1
RAW_DIR = "../experiment1_yolo11_baseline/results/exp1_baseline/raw"

# Output directory for Experiment 2
OUTPUT_DIR = "results/exp2_finetuned"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
CSV_PATH = os.path.join(OUTPUT_DIR, "detections.csv")

# Confidence threshold
CONF_THRESHOLD = 0.50


# =========================
# 2. Create Output Directory
# =========================

os.makedirs(ANNOTATED_DIR, exist_ok=True)


# =========================
# 3. Load the Fine-Tuned Model
# =========================

model = YOLO(MODEL_PATH)

print(f"Loaded model: {MODEL_PATH}")


# =========================
# 4. Create CSV File
# =========================

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "image_id",
        "predicted_class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "conf_threshold"
    ])


    # =========================
    # 5. Test Each Image
    # =========================

    image_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for image_name in image_files:

        image_path = os.path.join(RAW_DIR, image_name)

        # Run YOLO inference
        results = model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # Save the image with predicted bounding boxes
        annotated_image = result.plot()

        annotated_path = os.path.join(
            ANNOTATED_DIR,
            image_name
        )

        cv2.imwrite(
            annotated_path,
            annotated_image
        )

        boxes = result.boxes

        # =========================
        # 6. No Object Detected
        # =========================

        if boxes is None or len(boxes) == 0:

            writer.writerow([
                image_name,
                "NO_DETECTION",
                "",
                "",
                "",
                "",
                "",
                CONF_THRESHOLD
            ])

            print(f"{image_name}: NO_DETECTION")

        # =========================
        # 7. Object Detected
        # =========================

        else:

            for box in boxes:

                class_id = int(box.cls[0])

                predicted_class = model.names[class_id]

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                writer.writerow([
                    image_name,
                    predicted_class,
                    round(confidence, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    CONF_THRESHOLD
                ])

                print(
                    f"{image_name}: "
                    f"{predicted_class} "
                    f"{confidence:.2f}"
                )


print("\nTesting completed.")
print(f"CSV saved to: {CSV_PATH}")
print(f"Annotated images saved to: {ANNOTATED_DIR}")