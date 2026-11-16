
CATEGORY_COMPATIBILITY = {
    "main_dish": {
        "sauce": 0.95,
        "side_dish": 0.90,
        "beverage": 0.85,
        "dessert": 0.20,
        "main_dish": 0.10,
    },
    "beverage": {
        "snack": 0.70,
        "dessert": 0.60,
    },
}


def compatibility_score(
    cart_category: str,
    candidate_category: str,
) -> float:
    return CATEGORY_COMPATIBILITY.get(
        cart_category, {}
    ).get(candidate_category, 0.0)
