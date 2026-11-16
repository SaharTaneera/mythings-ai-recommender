
"""Experimental cross-encoder baseline for complementary product ranking.

This module uses a multilingual passage-ranking CrossEncoder to score
candidate products jointly with a bilingual complementarity query.

The underlying model was pretrained for relevance ranking, not
specifically for complementary product classification. Its scores
are ranking signals, not calibrated complementarity probabilities.

This experimental implementation is not used by the final
Qwen2.5 + multilingual E5 recommendation pipeline.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

from mythings_ai_recommender.schemas.product import Product


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def get_pairwise_model() -> CrossEncoder:
    """Load the CrossEncoder once and reuse it across calls."""
    return CrossEncoder(MODEL_NAME)


def product_text(product: Product) -> str:
    """Combine available bilingual product metadata into one string."""
    parts = [
        product.name_en,
        product.name_ar,
        product.category,
        product.subcategory or "",
        product.description_en or "",
        product.description_ar or "",
    ]

    return " | ".join(part for part in parts if part)


def score_product_pairs(
    cart_product: Product,
    candidates: list[Product],
) -> list[tuple[str, float]]:
    """Rank candidate products against a cart product.

    Args:
        cart_product: Product currently in the cart.
        candidates: Eligible products to rank.

    Returns:
        A list of (product_id, score) tuples sorted by descending score.
        Scores are model-generated ranking signals, not probabilities.
    """
    if not candidates:
        return []

    cart_text = product_text(cart_product)

    # Frame the query as a complementary-product retrieval task.
    query = (
        "Find a suitable complementary add-on to: "
        f"{cart_text}. "
        "ابحث عن منتج إضافي مناسب مع هذا المنتج."
    )

    pairs = [
        (query, product_text(candidate))
        for candidate in candidates
    ]

    # Batch prediction avoids one model call per candidate.
    scores = get_pairwise_model().predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    results = [
        (candidate.id, float(score))
        for candidate, score in zip(
            candidates,
            np.ravel(scores),
        )
    ]

    return sorted(
        results,
        key=lambda result: result[1],
        reverse=True,
    )
