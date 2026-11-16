
from mythings_ai_recommender.models.complementarity import (
    semantic_pairing_score,
)


candidates = [
    "Garlic sauce for chicken shawarma",
    "Chocolate dessert sauce",
    "Pepsi soft drink",
    "French fries with salt",
    "Beef shawarma sandwich",
    "ثومية للشاورما",
    "صوص شوكولاتة للحلويات",
    "بطاطا مقلية",
]


results = []

for candidate in candidates:
    score = semantic_pairing_score(
        cart_category="main_dish",
        candidate_text=candidate,
    )

    results.append((candidate, score))


results.sort(
    key=lambda item: item[1],
    reverse=True,
)


print("\nSemantic pairing experiment:\n")

for name, score in results:
    print(f"{score:.4f}  {name}")
