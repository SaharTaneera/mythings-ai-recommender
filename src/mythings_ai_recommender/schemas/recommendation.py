
from pydantic import BaseModel, Field

from mythings_ai_recommender.schemas.cart import Cart
from mythings_ai_recommender.schemas.product import Product


class RecommendationRequest(BaseModel):
    cart: Cart

    
    products: list[Product] = Field(default_factory=list)


    candidates: list[Product] = Field(
        default_factory=list
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )

class RecommendedProduct(BaseModel):
    product_id: str
    score: float = Field(ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedProduct]
    model_version: str
