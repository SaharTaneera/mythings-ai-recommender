
import pytest
from fastapi.testclient import TestClient

from mythings_ai_recommender.main import app
from mythings_ai_recommender.services import recommender


client = TestClient(app)


class FakeRanker:
    """Predictable ranker for testing API behavior."""

    def rank(self, cart_products, candidates):
        return [
            (product.id, 0.90 - index * 0.05)
            for index, product in enumerate(candidates)
        ]


@pytest.fixture(autouse=True)
def mock_ai_models(monkeypatch):
    """
    Isolate API tests from real Qwen and E5 inference.

    The recommendation filter is replaced with a pass-through,
    so these tests focus on API and business logic.
    """

    monkeypatch.setattr(
        recommender,
        "get_ranker",
        lambda: FakeRanker(),
    )

    monkeypatch.setattr(
        recommender,
        "filter_complementary_candidates",
        lambda cart_products, candidates: candidates,
    )


def make_product(
    product_id,
    merchant_id="R001",
    name="Test Product",
):
    return {
        "id": product_id,
        "merchant_id": merchant_id,
        "name_en": name,
    }


def make_request(cart_ids, candidates, top_k=3):
    return {
        "cart": {
            "merchant_id": "R001",
            "items": [
                {"product_id": product_id, "quantity": 1}
                for product_id in cart_ids
            ],
        },
        "products": [
            make_product(product_id)
            for product_id in cart_ids
        ],
        "candidates": candidates,
        "top_k": top_k,
    }


def test_excludes_cart_items():
    payload = make_request(
        cart_ids=["101"],
        candidates=[
            make_product("101", name="Shawarma"),
            make_product("102", name="Garlic Sauce"),
        ],
    )

    response = client.post(
        "/api/v1/recommendations",
        json=payload,
    )

    assert response.status_code == 200

    ids = [
        item["product_id"]
        for item in response.json()["recommendations"]
    ]

    assert ids == ["102"]


def test_excludes_other_merchants():
    payload = make_request(
        cart_ids=["101"],
        candidates=[
            make_product("102", merchant_id="R001"),
            make_product("103", merchant_id="R002"),
        ],
    )

    response = client.post(
        "/api/v1/recommendations",
        json=payload,
    )

    assert response.status_code == 200

    ids = [
        item["product_id"]
        for item in response.json()["recommendations"]
    ]

    assert ids == ["102"]


def test_missing_metadata_and_top_k():
    payload = make_request(
        cart_ids=["101"],
        candidates=[
            {
                "id": "102",
                "merchant_id": "R001",
                "name_ar": "ثومية",
            },
            {
                "id": "103",
                "merchant_id": "R001",
                "name_en": "Pepsi",
            },
        ],
        top_k=1,
    )

    response = client.post(
        "/api/v1/recommendations",
        json=payload,
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 1
    assert response.json()["recommendations"][0]["product_id"] == "102"


def test_empty_cart_returns_no_recommendations():
    payload = make_request(
        cart_ids=[],
        candidates=[make_product("102")],
    )

    response = client.post(
        "/api/v1/recommendations",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_rejects_product_without_a_name():
    payload = make_request(
        cart_ids=["101"],
        candidates=[
            {
                "id": "102",
                "merchant_id": "R001",
            }
        ],
    )

    response = client.post(
        "/api/v1/recommendations",
        json=payload,
    )

    assert response.status_code == 422
