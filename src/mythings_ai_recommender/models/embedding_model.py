
"""Shared multilingual E5 embedding utilities.

Provides a cached SentenceTransformer model and a helper for
encoding product descriptions into normalized embedding vectors.

The pretrained model is used for inference; no fine-tuning
is performed in this module.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and reuse the multilingual E5 model within this process."""
    return SentenceTransformer(MODEL_NAME)


def embed_product(text: str) -> np.ndarray:
    """Encode product text as a normalized E5 passage embedding.

    Args:
        text: Product name or descriptive text.

    Returns:
        A one-dimensional NumPy embedding vector with unit length.
    """
    model = get_embedding_model()

    # E5 distinguishes passages from queries using input prefixes.
    embedding = model.encode(
        f"passage: {text}",
        normalize_embeddings=True,
    )

    return embedding
