"""Evaluate the local Qwen relationship classifier on a JSON benchmark.

Usage:
    uv run python evaluate_relationships_dataset.py --split test
    uv run python evaluate_relationships_dataset.py --split dev

All reference labels are synthetic proposals, not verified MyThings ground truth.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mythings_ai_recommender.models.relationship_classifier import (
    classify_one_relationship,
)
from mythings_ai_recommender.schemas.product import Product

LABELS = ["complement", "substitute", "unrelated", "uncertain"]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()

    path = Path("data/evaluation") / f"relationship_{args.split}.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No evaluation cases in {path}")

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation case IDs")
    if any(case["expected"] not in LABELS for case in cases):
        raise ValueError("Unknown label in evaluation set")

    print(f"Evaluating {len(cases)} synthetic {args.split} cases from {path}")
    print("Keep the classifier prompt/model frozen while evaluating.\n")

    output = []
    for i, case in enumerate(cases, 1):
        # Same input format used by our classifier; names may be Arabic.
        cart_product = Product(id=f"cart-{case['id']}", merchant_id="EVAL",
                               name_en=case["cart_product"])
        candidate = Product(id=f"candidate-{case['id']}", merchant_id="EVAL",
                            name_en=case["candidate"])

        start = perf_counter()
        predicted = classify_one_relationship([cart_product], candidate)
        elapsed = perf_counter() - start
        if predicted not in LABELS:
            raise ValueError(f"Unexpected model output: {predicted!r}")

        row = {**case, "predicted": predicted, "correct": predicted == case["expected"],
               "latency_seconds": round(elapsed, 4)}
        output.append(row)
        print(f"{i:03}/{len(cases)} {'PASS' if row['correct'] else 'FAIL'} "
              f"{case['cart_product']} -> {case['candidate']} "
              f"[{case['expected']} / {predicted}] {elapsed:.2f}s", flush=True)

    y_true = [r["expected"] for r in output]
    y_pred = [r["predicted"] for r in output]
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("Macro-F1 (3 substantive classes):",
          f"{f1_score(y_true, y_pred, labels=LABELS[:3], average='macro', zero_division=0):.3f}")
    print("Macro-F1 (all 4 classes):",
          f"{f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.3f}")
    binary_true = [x == "complement" for x in y_true]
    binary_pred = [x == "complement" for x in y_pred]
    p, r, f, _ = precision_recall_fscore_support(binary_true, binary_pred,
                                                  average="binary", zero_division=0)
    print(f"Complement precision={p:.3f}, recall={r:.3f}, F1={f:.3f}")
    print("Confusion matrix rows=actual, cols=predicted; labels:", LABELS)
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    for field in ("domain", "language"):
        print(f"\nAccuracy by {field}:")
        for value in sorted({row[field] for row in output}):
            subset = [row for row in output if row[field] == value]
            print(f"  {value}: {sum(row['correct'] for row in subset)}/{len(subset)}")

    timings = [row["latency_seconds"] for row in output]
    print(f"\nObserved mean latency={mean(timings):.3f}s; "
          f"median={median(timings):.3f}s; P95={percentile(timings, .95):.3f}s")
    print("NOTE: timings can include model warm-up and cached responses; not production P95.")

    result_path = Path("data/evaluation") / f"relationship_{args.split}_predictions.csv"
    with result_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"\nSaved predictions: {result_path}")
    print("IMPORTANT: Synthetic proposed labels; review disagreements and ambiguous pairs, "
          "and report this as a small synthetic benchmark, not MyThings production accuracy.")


if __name__ == "__main__":
    main()
