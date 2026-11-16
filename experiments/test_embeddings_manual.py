
from mythings_ai_recommender.models.embedding_model import (
    embed_product,
)

products = [
    "Chicken Shawarma",
    "شاورما دجاج",
    "Garlic Sauce",
    "Chocolate Sauce",
]

for product in products:
    vector = embed_product(product)

    print(
        f"{product}: "
        f"embedding shape = {vector.shape}"
    )
