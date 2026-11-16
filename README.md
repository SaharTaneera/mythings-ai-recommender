# MyThings AI Recommender

A **local, multilingual, cart-aware complementary-product recommendation prototype** built for a multi-merchant shopping use case. The prototype uses a Python FastAPI service, locally hosted Qwen2.5 7B through Ollama, multilingual E5 embeddings, and a React/TypeScript demo.

> **Scope:** Proof of concept using synthetic catalog data and manually labeled evaluation examples. Not integrated with the live MyThings platform, inventory, payments, or real customer history.

## What it does

Given a merchant's shopping cart and eligible candidate products, the service suggests add-ons that complement items already in the cart rather than alternatives or unrelated products. It supports English, Arabic, and bilingual product metadata.

1. **Eligibility:** Keep candidates from the cart's merchant, exclude products already in the cart, and deduplicate candidate IDs.
2. **Relationship filtering:** Qwen2.5 7B evaluates each cart-product/candidate pair as `complement`, `substitute`, `unrelated`, or `uncertain`. A candidate is retained if it complements at least one cart item.
3. **Ranking:** `intfloat/multilingual-e5-small` compares candidates against bilingual positive/negative pairing prompts and sorts the accepted items using a heuristic semantic score.
4. **Response:** Return up to `top_k` product IDs, scores, and a model-version identifier.

The score is a **ranking heuristic, not a probability**. When fewer than `top_k` items qualify, fewer are returned.

## Architecture

```text
React/TypeScript demo (synthetic catalog)
             |
             v
POST /api/v1/recommendations
             |
             v
FastAPI + Pydantic validation
             |
             v
Merchant/cart eligibility filtering
             |
             v
Qwen2.5 7B (Ollama, pairwise, process-local cached classification)
             |
             v
Multilingual E5 semantic reranking
             |
             v
Top-K product IDs and heuristic scores
```

In an intended commerce integration, the host application (for example, a .NET backend) owns catalog data, stock, prices, authorization, and business eligibility. It supplies permitted candidates to the AI service and revalidates recommendations before display. **No live .NET integration is claimed.**

## Choose how to run

**Option A — Docker Compose (recommended for reviewers):** Docker packages the Python API and Ollama as separate services, so Ollama does **not** need to be installed separately on the host. Qwen still must be downloaded **inside** the container, and sufficient hardware is needed.

```powershell
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b
```

Then open `http://127.0.0.1:8000/docs`. The React demo can be started separately from `frontend/` using `npm install` and `npm run dev`.

See [Docker setup and limitations](docs/DOCKER.md). The Compose deployment is a demonstration; it does not configure GPU acceleration or production security.

**Option B — Manual local installation:** Use Python/uv with an Ollama instance installed on the host, following the instructions below.

For architectural rationale, experimental baselines, actual benchmark scope, tradeoffs, and future improvements, read [Engineering Decisions](docs/ENGINEERING_DECISIONS.md). A [C#/.NET HTTP client example](integration/dotnet/README.md) demonstrates the intended commerce integration boundary; **it is illustrative, not live integration**.

## Prerequisites

- Python **3.12+**, [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally, with `qwen2.5:7b` downloaded
- Node.js and npm for the frontend
- Enough RAM/VRAM to run Qwen2.5 7B; first embedding-model use may download model files

## Run locally

In the repository root:

```powershell
ollama pull qwen2.5:7b
uv sync
uv run uvicorn mythings_ai_recommender.main:app --reload --host 127.0.0.1 --port 8000
```

Start the Ollama service separately if it is not running already. Check the API at `http://127.0.0.1:8000/docs` and `http://127.0.0.1:8000/health`.

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. Vite proxies `/api` to `127.0.0.1:8000`. The frontend uses **demonstration products and prices only**.

## API example

`POST /api/v1/recommendations`

```json
{
  "cart": {
    "merchant_id": "R001",
    "items": [{"product_id": "101", "quantity": 1}]
  },
  "products": [
    {"id": "101", "merchant_id": "R001", "name_en": "Chicken Shawarma", "category": "main_dish"}
  ],
  "candidates": [
    {"id": "102", "merchant_id": "R001", "name_en": "Garlic Sauce", "category": "sauce"},
    {"id": "103", "merchant_id": "R001", "name_ar": "بطاطا مقلية", "category": "side_dish"}
  ],
  "top_k": 3
}
```

Successful responses contain `recommendations: [{product_id, score}]` and `model_version`. Actual recommendations vary with local model inference.

## Evaluation

Run tests:

```powershell
uv run pytest tests/ -v
```

Run benchmark scripts **from the repository root**:

```powershell
uv run python -m evaluation.evaluate_relationships_dataset --split dev
uv run python -m evaluation.evaluate_relationships_dataset --split test
uv run python -m evaluation.evaluate_ranking
```

**Previously observed results (local runs; not reproduced by the GitHub review):**

| Evaluation | Observed result |
|---|---|
| 20-example development classification | 90% accuracy |
| 100-example synthetic relationship classification | 81% accuracy; complement precision 79.6%, recall 97.5%, F1 87.6% |
| Six development cart scenarios: E5-only | Precision@3 55.6%, NDCG@3 50.0% |
| Six development cart scenarios: Qwen-only | Precision@3 100%, NDCG@3 100% |
| Six development cart scenarios: Qwen + E5 | Precision@3 100%, NDCG@3 100% |
| Mocked FastAPI unit tests | 5/5 passed |

The six cart scenarios contain exactly three binary-labeled relevant candidates each, so they **cannot establish that E5 reranking improves over Qwen-only**. Qwen-only preserves input catalog order. The relationship benchmark uses proposed synthetic labels, not verified MyThings ground truth. The classifier's substitute recall was 50% on the 100-case set. Benchmark latency reflects one local environment and may include caching and warm-up; it is not production performance.

Dataset files and recorded predictions are in `data/evaluation/`; scripts are in `evaluation/`.

## Repository layout

- `src/mythings_ai_recommender/`: API, Pydantic schemas, active Qwen classifier, E5 model and ranking service
- `frontend/`: React + Vite bilingual demo
- `evaluation/`: classification and three-way ranking benchmark scripts
- `data/evaluation/`: synthetic evaluation sets and recorded outputs
- `data/training/`: synthetic examples for experimental training
- `experiments/`: earlier baselines and cross-encoder experiments; **not part of the API inference path**
- `artifacts/`: files from a previous experimental model run; not needed to run the final API
- `tests/`: mocked API/business-logic tests
- `scripts/`: dataset-inspection utilities

## Limitations and next steps

- Qwen is called per cart-item/candidate pair; cost and latency scale with candidate count. A retrieval shortlist, batching, and operational benchmarking are needed for larger catalogs.
- Cached outcomes are process-local; inference failures currently map to `uncertain` and can be cached.
- The multi-item policy accepts any candidate complementary to at least one cart item, even if it substitutes another item.
- The service does not validate stock, catalog permissions, prices, or current merchant availability. Its `/health` endpoint reports service liveness, not Ollama readiness.
- The current evaluation set is small, synthetic, and not independent real shopping data. Further work should include reviewed Arabic/Jordanian labels, held-out merchants, graded ranking judgments, and real basket/feedback analysis.
- E5 scores and ranking weights are not learned from purchases and do not represent calibrated purchase likelihood.
- No production authentication, monitoring, deployment, or end-to-end MyThings integration is provided.

## Development provenance

Source code, evaluation scripts, and earlier experiments are retained to make the approach inspectable. The final prototype uses **pretrained models and prompt-based classification**, not a claimed fine-tuned Qwen classifier. Experimental cross-encoder training is separate from the final service.
