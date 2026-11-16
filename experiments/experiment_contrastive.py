
import numpy as np

from mythings_ai_recommender.models.embedding_model import (
    get_embedding_model,
)

model = get_embedding_model()

positive_examples = [
    "A savory garlic dipping sauce served with shawarma",
    "A side dish of fries served with shawarma",
    "A soft drink accompanying a shawarma meal",
]

negative_examples = [
    "A sweet chocolate topping for desserts",
    "Another main dish such as a shawarma sandwich",
    "A sweet sauce intended for cakes and ice cream",
]

candidates = [
    "Garlic sauce",
    "Chocolate dessert sauce",
    "Pepsi soft drink",
    "French fries",
    "Beef shawarma sandwich",
    "ثومية",
    "صوص شوكولاتة",
    "بطاطا مقلية",
]

positive_vectors = model.encode(
    [f"query: {text}" for text in positive_examples],
    normalize_embeddings=True,
)

negative_vectors = model.encode(
    [f"query: {text}" for text in negative_examples],
    normalize_embeddings=True,
)

candidate_vectors = model.encode(
    [f"passage: {text}" for text in candidates],
    normalize_embeddings=True,
)

results = []

for name, vector in zip(candidates, candidate_vectors):

    positive_score = float(
        np.max(positive_vectors @ vector)
    )

    negative_score = float(
        np.max(negative_vectors @ vector)
    )

    final_score = positive_score - negative_score

    results.append((name, final_score))

results.sort(
    key=lambda item: item[1],
    reverse=True,
)

for name, score in results:
    print(f"{score:+.4f}  {name}")
