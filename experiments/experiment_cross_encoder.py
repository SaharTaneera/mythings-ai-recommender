
from experiments.cross_encoder.pairwise_ranker import (
    score_product_pairs,
)
from mythings_ai_recommender.schemas.product import Product


def product(
    id: str,
    name_en: str,
    name_ar: str,
    category: str,
    description_en: str = "",
) -> Product:
    return Product(
        id=id,
        merchant_id="R001",
        name_en=name_en,
        name_ar=name_ar,
        category=category,
        description_en=description_en,
    )


cart_product = product(
    "101",
    "Chicken Shawarma",
    "شاورما دجاج",
    "main_dish",
)

candidates = [
    product(
        "102", "Garlic Sauce", "ثومية", "sauce",
        "Savory garlic dipping sauce for shawarma",
    ),
    product(
        "106", "Chocolate Sauce", "صوص شوكولاتة", "sauce",
        "Sweet chocolate topping for desserts",
    ),
    product(
        "103", "Pepsi", "بيبسي", "beverage",
    ),
    product(
        "104", "French Fries", "بطاطا مقلية", "side_dish",
    ),
    product(
        "107", "Beef Shawarma", "شاورما لحم", "main_dish",
    ),
]

names = {item.id: item.name_en for item in candidates}

results = score_product_pairs(cart_product, candidates)

print("\nCross-encoder results:\n")

for item_id, score in results:
    print(f"{item_id} | {names[item_id]:20} | {score:+.4f}")
