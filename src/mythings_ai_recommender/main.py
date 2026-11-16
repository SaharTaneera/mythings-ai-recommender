
from fastapi import FastAPI
from mythings_ai_recommender.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)

from mythings_ai_recommender.services.recommender import (
    recommend_products,
)

app = FastAPI(
    title="MyThings AI Recommender",
    description="Multilingual cart-aware recommendation service",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mythings-ai-recommender"
    }



@app.post(
    "/api/v1/recommendations",
    response_model=RecommendationResponse,
)
def recommend(request: RecommendationRequest):
    return recommend_products(request)


