
from mythings_ai_recommender.models.relationship_classifier import (
    classify_one_relationship,
)
from mythings_ai_recommender.schemas.product import Product


def product(product_id, en=None, ar=None, description=None):
    return Product(
        id=product_id,
        merchant_id="R001",
        name_en=en,
        name_ar=ar,
        description_en=description,
    )


cart = [
    product("101", "Beef Burger", "برجر لحم"),
]

candidates = [
    product("102", "Garlic Sauce", "ثومية"),
    product("103", "French Fries", "بطاطا مقلية"),
    product("104", "Pepsi", "بيبسي"),
    product(
        "105",
        "Chocolate Sauce",
        "صوص شوكولاتة",
        "Sweet chocolate topping for desserts",
    ),
    product("106", "Chicken Burger", "برجر دجاج"),
]

results = {
    candidate.id: classify_one_relationship(cart, candidate)
    for candidate in candidates
}

print("\nProduct relationship classifications:\n")

for candidate in candidates:
    name = candidate.name_en or candidate.name_ar
    print(f"{name:25} -> {results[candidate.id]}")
