import csv
import os

# Input and output CSV paths
INPUT_CSV = "results/exp2_finetuned/detections.csv"
OUTPUT_CSV = "results/exp2_finetuned/detections_with_ground_truth.csv"

# Ground truth labels for the 15 test images
GROUND_TRUTH = {
    "test01.jpg": "garlic",
    "test02.jpg": "garlic",
    "test03.jpg": "apple",
    "test04.jpg": "apple",
    "test05.jpg": "apple",
    "test06.jpg": "apple",
    "test07.jpg": "apple",
    "test08.jpg": "key",
    "test09.jpg": "key",
    "test10.jpg": "game_controller",
    "test11.jpg": "glasses",
    "test12.jpg": "glasses",
    "test13.jpg": "green_chili",
    "test14.jpg": "green_chili",
    "test15.jpg": "green_chili",
}

rows = []

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    fieldnames = reader.fieldnames + [
        "ground_truth",
        "evaluation"
    ]

    for row in reader:
        image_id = row["image_id"]
        predicted_class = row["predicted_class"]

        ground_truth = GROUND_TRUTH.get(image_id, "UNKNOWN")

        if predicted_class == "NO_DETECTION":
            evaluation = "miss"
        elif predicted_class == ground_truth:
            evaluation = "correct"
        else:
            evaluation = "wrong"

        row["ground_truth"] = ground_truth
        row["evaluation"] = evaluation

        rows.append(row)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print("Ground truth and evaluation added.")
print(f"Saved to: {OUTPUT_CSV}")