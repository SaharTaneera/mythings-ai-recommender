
"""Multilingual E5 semantic ranking for complementary products.

The ranker compares candidate products with bilingual positive
and negative pairing descriptions using normalized E5 embeddings.

In the final recommendation pipeline, Qwen2.5 7B first filters
candidates by relationship. E5 then ranks accepted candidates.

No manually defined category compatibility scores are required.
Product category metadata can still contribute to text embeddings.

Scores are heuristic ranking signals, not calibrated probabilities.
"""

import numpy as np

from mythings_ai_recommender.models.embedding_model import (
    get_embedding_model,
)
from mythings_ai_recommender.schemas.product import Product


def product_text(product: Product) -> str:
    """Combine available English and Arabic product metadata."""
    fields = [
        ("Product", product.name_en),
        ("المنتج", product.name_ar),
        ("Category", product.category),
        ("Type", product.subcategory),
        ("Description", product.description_en),
        ("الوصف", product.description_ar),
    ]

    return ". ".join(
        f"{label}: {value.strip()}"
        for label, value in fields
        if value and value.strip()
    )


def pairing_prompts(
    product: Product,
) -> tuple[list[str], list[str]]:
    """Create bilingual positive and negative pairing descriptions."""
    name = " / ".join(
        value.strip()
        for value in [product.name_en, product.name_ar]
        if value and value.strip()
    )

    context = product_text(product)

    positive = [
        (
            f"A useful complementary product to buy together with "
            f"{name}. It serves an additional purpose, works alongside "
            f"the original product, and helps complete its use. "
            f"Original product: {context}"
        ),
        (
            f"منتج مكمل ومفيد يتم شراؤه مع {name}، "
            f"يستخدم بجانبه أو يساعد في الاستفادة منه "
            f"بدلاً من استبداله. "
            f"المنتج الأصلي: {context}"
        ),
    ]

    negative = [
        (
            f"A duplicate, alternative, or replacement for {name}, "
            f"rather than a complementary product. "
            f"Original product: {context}"
        ),
        (
            f"منتج بديل أو مشابه أو مكرر للمنتج {name}، "
            f"وليس منتجاً مكملاً له. "
            f"المنتج الأصلي: {context}"
        ),
    ]

    return positive, negative


class HybridRanker:
    """Rank candidate products using bilingual E5 semantic signals.

    The existing class name is preserved for compatibility with
    recommender.py and evaluation scripts.
    """

    def __init__(self) -> None:
        # Reuse the cached E5 model instead of loading it per request.
        self.model = get_embedding_model()

    def rank(
        self,
        cart_products: list[Product],
        candidates: list[Product],
    ) -> list[tuple[str, float]]:
        """Return candidate IDs and heuristic scores, highest first."""
        if not cart_products or not candidates:
            return []

        # Encode candidate descriptions as E5 passages.
        candidate_texts = [
            product_text(candidate)
            for candidate in candidates
        ]

        candidate_vectors = self.model.encode(
            [f"passage: {text}" for text in candidate_texts],
            normalize_embeddings=True,
        )

        # Generate positive and negative pairing prompts for the cart.
        prompts = [
            pairing_prompts(product)
            for product in cart_products
        ]

        positive_texts = [
            text
            for positive, _ in prompts
            for text in positive
        ]

        negative_texts = [
            text
            for _, negative in prompts
            for text in negative
        ]

        # Encode prompts as E5 queries.
        positive_vectors = self.model.encode(
            [f"query: {text}" for text in positive_texts],
            normalize_embeddings=True,
        )

        negative_vectors = self.model.encode(
            [f"query: {text}" for text in negative_texts],
            normalize_embeddings=True,
        )

        # Matrix multiplication produces cosine similarity scores
        # because all embedding vectors have unit length.
        positive_scores = positive_vectors @ candidate_vectors.T
        negative_scores = negative_vectors @ candidate_vectors.T

        results = []

        for index, candidate in enumerate(candidates):
            # Compare the strongest positive and negative similarities.
            semantic_margin = (
                float(np.max(positive_scores[:, index]))
                - float(np.max(negative_scores[:, index]))
            )

            # Convert the margin to a bounded heuristic score.
            # This is NOT a calibrated probability.
            semantic_score = float(
                np.clip(
                    0.5 + 2.0 * semantic_margin,
                    0.0,
                    1.0,
                )
            )

            results.append(
                (candidate.id, round(semantic_score, 4))
            )

        # Sort by score descending, breaking ties by product ID.
        return sorted(
            results,
            key=lambda item: (-item[1], item[0]),
        )
