
"""Experimental E5-based semantic complementarity scoring.

This module estimates how closely a candidate product matches a
category-specific description of complementary products.

It uses cosine similarity between normalized embeddings. The score
is a semantic matching heuristic, not a trained or calibrated
probability of complementarity.
"""

import numpy as np

from mythings_ai_recommender.models.embedding_model import (
    get_embedding_model,
)


# Descriptions of commonly complementary product relationships.
# These are heuristic prompts rather than learned relationship labels.
PAIRING_PROMPTS = {
    "main_dish": (
        "A complementary side, savory sauce, "
        "or drink served with a main meal"
    ),
    "beverage": (
        "A snack or food item that pairs with a drink"
    ),
    "dessert": (
        "A drink or topping that complements dessert"
    ),
}


def semantic_pairing_score(
    cart_category: str,
    candidate_text: str,
) -> float:
    """Return a heuristic semantic pairing score in [0, 1].

    Args:
        cart_category: Category of the product already in the cart.
        candidate_text: Text describing a candidate product.

    Returns:
        Cosine similarity between the E5 query and passage embeddings,
        clipped to [0, 1]. Higher scores indicate greater semantic
        alignment with the pairing description, not necessarily
        actual product complementarity.
    """
    model = get_embedding_model()

    # Use a generic pairing description for unknown categories.
    pairing_text = PAIRING_PROMPTS.get(
        cart_category,
        "An item that complements another product",
    )

    # E5 uses different prefixes for query and passage embeddings.
    query_embedding = model.encode(
        f"query: {pairing_text}",
        normalize_embeddings=True,
    )

    candidate_embedding = model.encode(
        f"passage: {candidate_text}",
        normalize_embeddings=True,
    )

    # Dot product equals cosine similarity for normalized vectors.
    similarity = float(
        np.dot(query_embedding, candidate_embedding)
    )

    # Preserve the original scoring behavior.
    return max(0.0, min(1.0, similarity))
