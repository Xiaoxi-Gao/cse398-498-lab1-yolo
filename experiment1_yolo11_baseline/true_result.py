import csv

CSV_PATH = "results/exp1_baseline/detections.csv"

# Ground truth labels for each test image
GROUND_TRUTH = {
    "test01": "garlic",
    "test02": "garlic",

    "test03": "apple",
    "test04": "apple",
    "test05": "apple",
    "test06": "apple",
    "test07": "apple",

    "test08": "key",
    "test09": "key",

    "test10": "game_controller",

    "test11": "glasses",
    "test12": "glasses",

    "test13": "green_chili",
    "test14": "green_chili",
    "test15": "green_chili",
}


def evaluate_prediction(predicted_class, ground_truth):
    if predicted_class == "NO_DETECTION":
        return "miss"

    if predicted_class == ground_truth:
        return "correct"

    return "wrong"


with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


for row in rows:
    image_id = row["image_id"]
    predicted_class = row["predicted_class"]

    ground_truth = GROUND_TRUTH.get(image_id, "UNKNOWN")

    row["ground_truth"] = ground_truth
    row["evaluation"] = evaluate_prediction(
        predicted_class,
        ground_truth
    )


fieldnames = rows[0].keys()

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"Updated {len(rows)} rows.")
print("Ground truth and evaluation added.")