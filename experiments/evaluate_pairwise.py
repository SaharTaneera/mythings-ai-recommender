
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


MODEL_PATH = "artifacts/pairwise_complementarity_v1"
EVAL_PATH = Path("data/eval_pairs.json")


def precision_at_k(ranked, k=2):
    return sum(label for _, label, _ in ranked[:k]) / k


def ndcg_at_k(ranked, k=2):
    dcg = sum(
        label / math.log2(index + 2)
        for index, (_, label, _) in enumerate(ranked[:k])
    )

    ideal = sorted(
        (label for _, label, _ in ranked),
        reverse=True,
    )[:k]

    idcg = sum(
        label / math.log2(index + 2)
        for index, label in enumerate(ideal)
    )

    return dcg / idcg if idcg > 0 else 0.0


def main():
    records = json.loads(
        EVAL_PATH.read_text(encoding="utf-8")
    )

    print("Loading fine-tuned model...")
    model = CrossEncoder(MODEL_PATH)

    pairs = [
        (row["cart_product"], row["candidate"])
        for row in records
    ]

    # The saved single-output CrossEncoder normally applies
    # sigmoid during predict(). These are ranking scores,
    # not calibrated purchase probabilities.
    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )
    ).reshape(-1)

    groups = defaultdict(list)

    for record, score in zip(records, scores):
        key = (
            record["domain"],
            record["cart_product"],
        )

        groups[key].append(
            (
                record["candidate"],
                int(record["label"]),
                float(score),
            )
        )

    precision_scores = []
    ndcg_scores = []

    for (domain, cart), items in groups.items():
        ranked = sorted(
            items,
            key=lambda row: row[2],
            reverse=True,
        )

        p2 = precision_at_k(ranked)
        n2 = ndcg_at_k(ranked)

        precision_scores.append(p2)
        ndcg_scores.append(n2)

        print(f"\nDOMAIN: {domain}")
        print(f"CART: {cart}")

        for candidate, label, score in ranked:
            status = "GOOD" if label == 1 else "BAD"

            print(
                f"  {score:.4f} | {status:4} | {candidate}"
            )

        print(f"Precision@2: {p2:.3f}")
        print(f"NDCG@2:      {n2:.3f}")

    print("\nOVERALL EVALUATION")
    print("-" * 45)
    print(
        f"Mean Precision@2: "
        f"{np.mean(precision_scores):.3f}"
    )
    print(
        f"Mean NDCG@2: "
        f"{np.mean(ndcg_scores):.3f}"
    )


if __name__ == "__main__":
    main()
