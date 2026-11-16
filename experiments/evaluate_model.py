
from mythings_ai_recommender.models.hybrid_ranker import (
    HybridRanker,
)
from mythings_ai_recommender.schemas.product import Product


def make_product(id, name_en=None, name_ar=None, description=None):
    return Product(
        id=id,
        merchant_id="R001",
        name_en=name_en,
        name_ar=name_ar,
        description_en=description,
    )


TEST_CASES = [
    {
        "name": "Food - Chicken Shawarma",
        "cart": make_product("101", "Chicken Shawarma", "شاورما دجاج"),
        "candidates": [
            make_product("102", "Garlic Sauce", "ثومية",
                         "Savory garlic sauce for shawarma"),
            make_product("103", "Pepsi", "بيبسي"),
            make_product("104", "French Fries", "بطاطا مقلية"),
            make_product("105", "Beef Shawarma", "شاورما لحم"),
            make_product("106", "Chocolate Sauce", "صوص شوكولاتة",
                         "Sweet topping for desserts"),
        ],
        "relevant": {"102", "103", "104"},
    },
    {
        "name": "Electronics - Smartphone",
        "cart": make_product("201", "USB-C Smartphone", "هاتف ذكي"),
        "candidates": [
            make_product("202", "USB-C Phone Charger", "شاحن هاتف"),
            make_product("203", "Compatible Phone Case", "غطاء هاتف"),
            make_product("204", "Another Smartphone", "هاتف آخر"),
            make_product("205", "Chocolate Bar", "لوح شوكولاتة"),
            make_product("206", "USB-C Charging Cable", "سلك شحن"),
        ],
        "relevant": {"202", "203", "206"},
    },
    {
        "name": "Personal Care - Shampoo",
        "cart": make_product("301", "Hair Shampoo", "شامبو للشعر"),
        "candidates": [
            make_product("302", "Hair Conditioner", "بلسم للشعر"),
            make_product("303", "Hair Mask", "ماسك للشعر"),
            make_product("304", "Another Shampoo", "شامبو آخر"),
            make_product("305", "Phone Charger", "شاحن هاتف"),
            make_product("306", "Hair Serum", "سيروم للشعر"),
        ],
        "relevant": {"302", "303", "306"},
    },
]


def precision_at_k(ranked_ids, relevant_ids, k=3):
    selected = ranked_ids[:k]
    hits = sum(item in relevant_ids for item in selected)
    return hits / k


def ndcg_at_k(ranked_ids, relevant_ids, k=3):
    import math

    dcg = sum(
        (1.0 / math.log2(index + 2))
        for index, item in enumerate(ranked_ids[:k])
        if item in relevant_ids
    )

    ideal_hits = min(k, len(relevant_ids))
    ideal_dcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


ranker = HybridRanker()

precision_scores = []
ndcg_scores = []

for case in TEST_CASES:
    ranked = ranker.rank(
        cart_products=[case["cart"]],
        candidates=case["candidates"],
    )

    names = {
        product.id: product.name_en or product.name_ar
        for product in case["candidates"]
    }

    ranked_ids = [product_id for product_id, _ in ranked]

    p3 = precision_at_k(ranked_ids, case["relevant"])
    n3 = ndcg_at_k(ranked_ids, case["relevant"])

    precision_scores.append(p3)
    ndcg_scores.append(n3)

    print(f"\n{case['name']}")
    print("-" * 50)

    for position, (product_id, score) in enumerate(ranked, 1):
        mark = "GOOD" if product_id in case["relevant"] else "BAD"
        print(
            f"{position}. {names[product_id]:25} "
            f"{score:.4f}  {mark}"
        )

    print(f"Precision@3: {p3:.3f}")
    print(f"NDCG@3:      {n3:.3f}")

print("\nOVERALL RESULTS")
print("-" * 50)
print(f"Mean Precision@3: {sum(precision_scores)/len(precision_scores):.3f}")
print(f"Mean NDCG@3:      {sum(ndcg_scores)/len(ndcg_scores):.3f}")
