
"""Orchestrate complementary-product recommendations.

Pipeline:
    1. Resolve cart products from request-provided catalog metadata.
    2. Filter candidates by merchant, cart membership, and duplicate ID.
    3. Keep products classified as complementary by Qwen2.5 7B.
    4. Rank accepted candidates using multilingual E5 semantic signals.
    5. Return the requested number of product IDs and heuristic scores.

This service does not manage catalog storage, pricing, or inventory.
Those responsibilities remain with the calling commerce application.
"""

from functools import lru_cache

from mythings_ai_recommender.models.hybrid_ranker import HybridRanker
from mythings_ai_recommender.models.relationship_classifier import (
    filter_complementary_candidates,
)
from mythings_ai_recommender.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
)


MODEL_VERSION = "qwen2.5-7b-pairwise-e5-v1"


@lru_cache(maxsize=1)
def get_ranker() -> HybridRanker:
    """Initialize the E5 ranker once and reuse it within the process."""
    return HybridRanker()


def recommend_products(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Generate complementary recommendations for a shopping cart.

    Args:
        request: Cart information, product metadata, candidate products,
            and the requested top-K recommendation count.

    Returns:
        Product IDs ranked by heuristic score, together with
        the model version. Returns an empty recommendation list
        when no eligible or complementary candidates are found.
    """
    merchant_id = request.cart.merchant_id

    # Track products already in the cart so they cannot be recommended.
    cart_ids = {
        item.product_id
        for item in request.cart.items
    }

    # Resolve cart product metadata for the current merchant.
    cart_products = [
        product
        for product in request.products
        if product.id in cart_ids
        and product.merchant_id == merchant_id
    ]

    # Build the eligible candidate pool.
    # Exclude products from other merchants, existing cart products,
    # and duplicate candidate IDs.
    candidates = []
    seen_ids = set()

    for candidate in request.candidates:
        if candidate.merchant_id != merchant_id:
            continue

        if candidate.id in cart_ids:
            continue

        if candidate.id in seen_ids:
            continue

        seen_ids.add(candidate.id)
        candidates.append(candidate)

    # Avoid unnecessary model inference for empty inputs.
    if not cart_products or not candidates:
        return RecommendationResponse(
            recommendations=[],
            model_version=MODEL_VERSION,
        )

    # Stage 1: Pairwise Qwen classification.
    # Only candidates predicted as complementary are retained.
    candidates = filter_complementary_candidates(
        cart_products=cart_products,
        candidates=candidates,
    )

    if not candidates:
        return RecommendationResponse(
            recommendations=[],
            model_version=MODEL_VERSION,
        )

    # Stage 2: Rank accepted complements with multilingual E5.
    ranked = get_ranker().rank(
        cart_products=cart_products,
        candidates=candidates,
    )

    # Return only product identifiers and heuristic ranking scores.
    # The calling application enriches these IDs with current prices,
    # inventory, and display metadata.
    return RecommendationResponse(
        recommendations=[
            RecommendedProduct(
                product_id=product_id,
                score=score,
            )
            for product_id, score in ranked[:request.top_k]
        ],
        model_version=MODEL_VERSION,
    )
