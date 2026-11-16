import csv
from math import log2
from pathlib import Path
from statistics import mean
from time import perf_counter

from mythings_ai_recommender.models.hybrid_ranker import HybridRanker
from mythings_ai_recommender.models.relationship_classifier import (
    filter_complementary_candidates,
)
from mythings_ai_recommender.schemas.product import Product


TOP_K = 3
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def product(pid, name, category=None, name_ar=None, description=None):
    return Product(
        id=pid,
        merchant_id="EVAL",
        name_en=name,
        name_ar=name_ar,
        category=category,
        description_en=description,
    )


# Hand-authored, provisional relevance judgments.
# These are development scenarios, not MyThings customer data.
SCENARIOS = [
    {
        "name": "Burger meal",
        "cart": [
            product("c1", "Beef Burger", "main_dish"),
        ],
        "candidates": [
            product("a1", "French Fries", "side_dish"),
            product("a2", "Garlic Sauce", "sauce"),
            product("a3", "Pepsi", "beverage"),
            product(
                "a4", "Chocolate Dessert Sauce", "sauce",
                description="Sweet chocolate topping for desserts",
            ),
            product("a5", "Chicken Burger", "main_dish"),
        ],
        "relevant": {"a1", "a2", "a3"},
    },
    {
        "name": "Mixed restaurant cart",
        "cart": [
            product("c2", "Chicken Shawarma", "main_dish"),
            product(
                "c3", "Chocolate Dessert Sauce", "dessert_topping"
            ),
        ],
        "candidates": [
            product("b1", "French Fries", "side_dish"),
            product("b2", "Garlic Sauce", "sauce"),
            product("b3", "Cola", "beverage"),
            product("b4", "Beef Shawarma", "main_dish"),
            product(
                "b5", "Vanilla Cake Frosting", "dessert_topping"
            ),
        ],
        "relevant": {"b1", "b2", "b3"},
    },
    {
        "name": "Pasta dinner",
        "cart": [
            product("c4", "Dry Spaghetti", "pasta"),
        ],
        "candidates": [
            product("d1", "Tomato Pasta Sauce", "sauce"),
            product("d2", "Parmesan Cheese", "cheese"),
            product("d3", "Fresh Basil", "herbs"),
            product(
                "d4", "Chocolate Dessert Sauce", "dessert_topping"
            ),
            product("d5", "Another Brand of Spaghetti", "pasta"),
        ],
        "relevant": {"d1", "d2", "d3"},
    },
    {
        "name": "Hair care",
        "cart": [
            product("c5", "Shampoo", "hair_care"),
        ],
        "candidates": [
            product("e1", "Hair Conditioner", "conditioner"),
            product("e2", "Hair Mask", "hair_treatment"),
            product("e3", "Leave-in Conditioner", "conditioner"),
            product("e4", "Phone Charger", "electronics"),
            product("e5", "Another Brand of Shampoo", "shampoo"),
        ],
        "relevant": {"e1", "e2", "e3"},
    },
    {
        "name": "Laptop accessories",
        "cart": [
            product("c6", "14-inch Laptop", "laptop"),
        ],
        "candidates": [
            product("f1", "Laptop Sleeve for 14-inch Laptop", "accessory"),
            product("f2", "Adjustable Laptop Stand", "accessory"),
            product("f3", "Wireless Mouse", "accessory"),
            product("f4", "Another 14-inch Laptop", "laptop"),
            product("f5", "Shampoo", "personal_care"),
        ],
        "relevant": {"f1", "f2", "f3"},
    },
    {
        "name": "Arabic shawarma",
        "cart": [
            product("c7", None, "main_dish", name_ar="شاورما دجاج"),
        ],
        "candidates": [
            product("g1", None, "sauce", name_ar="ثومية"),
            product("g2", None, "side_dish", name_ar="بطاطا مقلية"),
            product("g3", None, "beverage", name_ar="بيبسي"),
            product("g4", None, "main_dish", name_ar="شاورما لحم"),
            product(
                "g5", None, "dessert_topping",
                name_ar="صوص شوكولاتة للحلويات",
            ),
        ],
        "relevant": {"g1", "g2", "g3"},
    },
]


def precision_at_k(ids, relevant, k=TOP_K):
    # Missing recommendation positions count as zero.
    return sum(pid in relevant for pid in ids[:k]) / k


def ndcg_at_k(ids, relevant, k=TOP_K):
    dcg = sum(
        int(pid in relevant) / log2(position + 2)
        for position, pid in enumerate(ids[:k])
    )

    ideal_count = min(k, len(relevant))

    idcg = sum(
        1 / log2(position + 2)
        for position in range(ideal_count)
    )

    return dcg / idcg if idcg else 0.0


def evaluate():
    ranker = HybridRanker()
    records = []

    model_names = [
        "E5 only",
        "Qwen only",
        "Qwen + E5",
    ]

    print("\nMYTHINGS THREE-MODEL RANKING EVALUATION")
    print("=" * 65)

    for scenario in SCENARIOS:
        name = scenario["name"]
        cart = scenario["cart"]
        candidates = scenario["candidates"]
        relevant = scenario["relevant"]

        lookup = {
            p.id: p.name_en or p.name_ar
            for p in candidates
        }

        print(f"\nSCENARIO: {name}", flush=True)

        # ----------------------------------------
        # MODEL 1: E5 only
        # ----------------------------------------
        start = perf_counter()

        e5_ranked = ranker.rank(
            cart_products=cart,
            candidates=candidates,
        )

        e5_time = perf_counter() - start

        e5_ids = [
            product_id
            for product_id, score in e5_ranked[:TOP_K]
        ]

        # ----------------------------------------
        # MODEL 2: Qwen only
        # ----------------------------------------
        start = perf_counter()

        accepted = filter_complementary_candidates(
            cart_products=cart,
            candidates=candidates,
        )

        qwen_time = perf_counter() - start

        # Qwen currently returns labels, not scores.
        # Preserve the input catalog order.
        qwen_ids = [
            p.id for p in accepted[:TOP_K]
        ]

        # ----------------------------------------
        # MODEL 3: Qwen + E5
        # ----------------------------------------
        start = perf_counter()

        hybrid_ranked = (
            ranker.rank(
                cart_products=cart,
                candidates=accepted,
            )
            if accepted
            else []
        )

        e5_rerank_time = perf_counter() - start

        hybrid_time = qwen_time + e5_rerank_time

        hybrid_ids = [
            product_id
            for product_id, score in hybrid_ranked[:TOP_K]
        ]

        # ----------------------------------------
        # Evaluate all three models
        # ----------------------------------------
        predictions = [
            ("E5 only", e5_ids, e5_time),
            ("Qwen only", qwen_ids, qwen_time),
            ("Qwen + E5", hybrid_ids, hybrid_time),
        ]

        for model_name, top_ids, elapsed in predictions:
            precision = precision_at_k(
                top_ids, relevant
            )

            ndcg = ndcg_at_k(
                top_ids, relevant
            )

            coverage = int(
                len(top_ids) == TOP_K
            )

            false_recommendations = sum(
                pid not in relevant
                for pid in top_ids
            )

            records.append({
                "scenario": name,
                "model": model_name,
                "precision_at_3": precision,
                "ndcg_at_3": ndcg,
                "coverage_at_3": coverage,
                "false_recommendations": false_recommendations,
                "latency_seconds": elapsed,
                "recommended_ids": "|".join(top_ids),
                "recommended_names": " | ".join(
                    lookup[pid] for pid in top_ids
                ),
            })

            print(f"\n  {model_name}", flush=True)

            if not top_ids:
                print("    No recommendations", flush=True)

            for position, pid in enumerate(
                top_ids, start=1
            ):
                status = (
                    "RELEVANT"
                    if pid in relevant
                    else "NOT RELEVANT"
                )

                print(
                    f"    {position}. "
                    f"{lookup[pid]} [{status}]",
                    flush=True,
                )

            print(
                f"    Precision@3: {precision:.3f}",
                flush=True,
            )
            print(
                f"    NDCG@3:      {ndcg:.3f}",
                flush=True,
            )
            print(
                f"    Coverage@3:  {coverage}",
                flush=True,
            )
            print(
                f"    Latency:     {elapsed:.2f}s",
                flush=True,
            )

    # ----------------------------------------
    # Save results
    # ----------------------------------------
    output_path = OUTPUT_DIR / "ranking_results.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    # ----------------------------------------
    # Final model comparison
    # ----------------------------------------
    print("\n" + "=" * 65)
    print("FINAL THREE-MODEL COMPARISON")
    print("=" * 65)

    for model_name in model_names:
        subset = [
            r for r in records
            if r["model"] == model_name
        ]

        print(f"\n{model_name}")

        print(
            "Mean Precision@3:",
            f"{mean(r['precision_at_3'] for r in subset):.3f}",
        )
        print(
            "Mean NDCG@3:",
            f"{mean(r['ndcg_at_3'] for r in subset):.3f}",
        )
        print(
            "Coverage@3:",
            f"{mean(r['coverage_at_3'] for r in subset):.3f}",
        )
        print(
            "Total false recommendations:",
            sum(
                r["false_recommendations"]
                for r in subset
            ),
        )
        print(
            "Mean latency:",
            f"{mean(r['latency_seconds'] for r in subset):.2f}s",
        )

    print(f"\nResults saved to: {output_path}")

    print(
        "\nIMPORTANT: These are small, hand-labeled "
        "development scenarios, not production metrics."
    )

    print(
        "Qwen-only ties use catalog order. "
        "The Qwen filter is executed once per scenario "
        "and shared with the hybrid evaluation."
    )

    print(
        "Latency may include caching and model warm-up."
    )


if __name__ == "__main__":
    evaluate()
