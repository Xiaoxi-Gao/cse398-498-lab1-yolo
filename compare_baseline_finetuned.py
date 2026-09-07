import os
import csv
from collections import Counter

# =========================
# 1. CSV Paths
# =========================

BASELINE_CSV = (
    "experiment1_yolo11_baseline/"
    "results/exp1_baseline/detections.csv"
)

FINETUNED_CSV = (
    "experiment2_finetuning/"
    "results/exp2_finetuned/detections_with_ground_truth.csv"
)


# =========================
# 2. Load Results
# =========================

def load_results(csv_path):
    results = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            # Normalize image IDs:
            # test01.jpg -> test01
            # test01     -> test01
            image_id = os.path.splitext(row["image_id"])[0]

            # Keep the first result for each image
            if image_id not in results:
                results[image_id] = row

            # If multiple detections exist for one image,
            # prefer the correct detection
            elif row.get("evaluation") == "correct":
                results[image_id] = row

    return results


baseline = load_results(BASELINE_CSV)
finetuned = load_results(FINETUNED_CSV)

all_images = sorted(set(baseline.keys()) | set(finetuned.keys()))


# =========================
# 3. Print Comparison Table
# =========================

print("\nBaseline vs Fine-tuned Comparison\n")

print(
    f"{'Image':<12}"
    f"{'Ground Truth':<18}"
    f"{'Baseline':<20}"
    f"{'Baseline Eval':<16}"
    f"{'Fine-tuned':<20}"
    f"{'Fine Eval':<12}"
)

print("-" * 98)

for image_id in all_images:

    b = baseline.get(image_id, {})
    f = finetuned.get(image_id, {})

    ground_truth = (
        f.get("ground_truth")
        or b.get("ground_truth")
        or "UNKNOWN"
    )

    baseline_prediction = b.get(
        "predicted_class",
        "N/A"
    )

    baseline_evaluation = b.get(
        "evaluation",
        "N/A"
    )

    finetuned_prediction = f.get(
        "predicted_class",
        "N/A"
    )

    finetuned_evaluation = f.get(
        "evaluation",
        "N/A"
    )

    print(
        f"{image_id:<12}"
        f"{ground_truth:<18}"
        f"{baseline_prediction:<20}"
        f"{baseline_evaluation:<16}"
        f"{finetuned_prediction:<20}"
        f"{finetuned_evaluation:<12}"
    )


# =========================
# 4. Summary Function
# =========================

def summarize(results):
    counts = Counter()

    for row in results.values():
        evaluation = row.get(
            "evaluation",
            ""
        )

        counts[evaluation] += 1

    total = len(results)

    correct = counts["correct"]
    wrong = counts["wrong"]
    miss = counts["miss"]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    return (
        total,
        correct,
        wrong,
        miss,
        accuracy
    )


# =========================
# 5. Calculate Summary
# =========================

(
    b_total,
    b_correct,
    b_wrong,
    b_miss,
    b_acc
) = summarize(baseline)

(
    f_total,
    f_correct,
    f_wrong,
    f_miss,
    f_acc
) = summarize(finetuned)


# =========================
# 6. Print Summary
# =========================

print("\nSummary")
print("-" * 60)

print(
    f"Baseline:   "
    f"{b_correct}/{b_total} correct | "
    f"{b_wrong} wrong | "
    f"{b_miss} miss | "
    f"Accuracy = {b_acc:.1f}%"
)

print(
    f"Fine-tuned: "
    f"{f_correct}/{f_total} correct | "
    f"{f_wrong} wrong | "
    f"{f_miss} miss | "
    f"Accuracy = {f_acc:.1f}%"
)

improvement = f_acc - b_acc

print(
    f"Improvement: "
    f"{improvement:+.1f} percentage points"
)


# =========================
# 7. Final Conclusion
# =========================

print("\nConclusion")
print("-" * 60)

print(
    f"The baseline YOLO11 model achieved "
    f"{b_acc:.1f}% accuracy."
)

print(
    f"The fine-tuned YOLO11 model achieved "
    f"{f_acc:.1f}% accuracy."
)

print(
    f"Fine-tuning improved accuracy by "
    f"{improvement:.1f} percentage points."
)