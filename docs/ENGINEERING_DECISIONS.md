# Engineering Design, Experiments, and Decision Record

**Project:** MyThings AI Recommender  
**Status:** Evaluated local proof of concept, not a production or live MyThings integration  
**Purpose:** Explain what was implemented, why the design was chosen, what was tested, what the measurements do and do not establish, and how this could evolve.

## 1. Problem and desired behavior

For a multi-merchant shopping platform, suggest **useful additional products** based on the contents of the current cart. The objective is **complementarity**, not merely topical similarity: garlic sauce may complement shawarma, while another shawarma is typically a substitute; dessert topping is not automatically suitable for a savory dish even if both are sauces.

Design goals:

- Multilingual support for **Arabic and English product titles/descriptions**.
- Merchant isolation: do not recommend from an unrelated seller.
- Exclude items already in the cart and deduplicate candidate IDs.
- Prioritize meaningful pairings over similarity alone; return zero results when nothing qualifies.
- Make the result consumable by an existing commerce backend rather than binding model inference directly to the frontend.
- Produce a demonstrable prototype without claiming access to MyThings catalog data, purchase logs, or live infrastructure.

Non-goals of this prototype: price optimization, collaborative personalization, inventory forecasting, checkout integration, conversion measurement, production-grade authentication, or trained purchase-probability prediction.

## 2. Delivered system and request workflow

```text
React demo OR future trusted .NET backend
         |
         | POST /api/v1/recommendations
         v
FastAPI + Pydantic request schema
         |
         | Resolve cart metadata supplied by caller
         | Restrict candidates to the cart merchant
         | Exclude cart IDs and duplicate candidates
         v
Qwen2.5 7B via Ollama
         |
         | Classify each cart-product / candidate pair:
         | complement | substitute | unrelated | uncertain
         | Keep candidate if it complements ANY cart item
         v
Multilingual E5 ("intfloat/multilingual-e5-small")
         |
         | Build bilingual positive and negative pairing prompts
         | Encode prompts as "query:" and candidates as "passage:"
         | Margin = best positive cosine - best negative cosine
         | Heuristic score = clip(0.5 + 2 * margin, 0, 1)
         v
Return top-K product IDs, scores, model version
         |
         v
Commerce layer resolves live price, stock, display details
```

The service expects the **calling application** to supply cart product metadata and a candidate list; there is no catalog database in the AI service. The API's `top_k` is 1–10 (default 3). The React demo uses a hardcoded synthetic catalog, sample prices, and bilingual labels.

### Example request behavior

Cart: Chicken Shawarma. Candidates: Garlic Sauce, French Fries, Beef Shawarma, Chocolate Dessert Sauce. The intended behavior is to keep relevant accompaniments, filter substitutes/unrelated candidates, then order retained items for display. Actual outputs vary with model inference and metadata.

## 3. Key engineering decisions

| Decision | Rationale | Tradeoff or limitation |
|---|---|---|
| **Model relationship explicitly** | A semantically similar alternative can be a poor add-on | Classification is imperfect, especially subtle substitutions |
| **Qwen2.5 7B prompted as a pairwise classifier** | Pretrained language reasoning can interpret sparse bilingual descriptions without domain-specific training data | Sequential model calls are expensive; zero-shot judgments may be inconsistent; no purchase-driven calibration |
| **Constrain outputs to four labels and parse JSON** | Explicit complement/substitute/unrelated/uncertain results simplify filtering and evaluation | A valid JSON response does not guarantee a correct semantic decision |
| **Filter first, rerank second** | Prevent obviously inappropriate candidates from appearing merely due to embedding similarity | Adds latency and more system complexity; current benchmark does **not** demonstrate incremental quality gain from the E5 reranker |
| **Multilingual E5 for ranking** | Lightweight multilingual semantic representations and bilingual prompt comparisons | Semantic scores depend on wording; prompt margin and score transformation are heuristic |
| **No active handcrafted category compatibility table** | Rules become brittle across restaurants, electronics, beauty, household, and new catalog types | Less deterministic than carefully maintained rules for a narrow domain |
| **FastAPI/Pydantic schema boundary** | Clearly testable HTTP contract with input validation and interactive docs | Prototype currently trusts upstream caller authorization and eligibility |
| **Merchant gating in service** | Basic defense against cross-merchant recommendations | Does not establish which products the caller is authorized to access |
| **Cache classification by product metadata** | Avoid repeated Qwen calls for identical pairs within one process | Process-local, no TTL; **failures mapped to uncertain are cached**, possibly hiding recovery |
| **Allow no-result responses** | Better than inventing unsuitable add-ons | No fallback recommendations or business-priority diversification yet |
| **Keep experimental methods in a separate folder** | Preserve evidence of exploration without mixing unused algorithms into the live inference path | Experimental checkpoints and training scripts may require additional assets |
| **Docker Compose with separate Ollama container** | CTO can start dependencies without independently installing Ollama | Initial model pull and significant hardware resources still required; reference deployment not production-hardened |
| **C# typed HTTP client example, not a replacement backend** | Demonstrates contract compatibility with an ASP.NET Core commerce application | Not integrated with real MyThings APIs, entities, credentials, or inventory |

The model classifier is a **pretrained Qwen2.5 7B prompted locally**, not a fine-tuned Qwen model. The historical cross-encoder fine-tuning experiment is a separate branch of the research work, not the active model.

## 4. Explored baselines and why they were not selected

### 4.1 Rule-based category compatibility

An early baseline used a table of category-pair scores (e.g., main dish with sauce or side dish). It is fast, interpretable, and can enforce business requirements. However, it requires hand maintenance, does not generalize to unseen categories, and can confuse a savory condiment with a sweet dessert sauce simply because both have the category `sauce`. Rules were removed from the active ranker, although merchant/cart eligibility checks remain deterministic.

**When to reconsider:** Mature catalog taxonomy, reliable complementary-bundle metadata, hard compatibility requirements, or a policy-first recommendation layer.

### 4.2 E5 embeddings alone

The E5-only approach can rank candidate descriptions by semantic association, is efficient, and does not require Qwen inference. However, proximity in embedding space is not the same as complementarity: substitutes and unrelated products can score well. In the recorded six-scenario comparison, E5-only generated eight false top-three recommendations.

**When to reconsider:** Use E5 (or a dedicated retrieval model) as a fast first-stage candidate retriever from a large catalog, followed by relationship verification.

### 4.3 Multilingual cross-encoder and experimental fine-tuning

Explored a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) and tiny synthetic labeled pairing sets. Cross-encoders can capture pair interactions more directly than independent embeddings, but the starting model targets general relevance; a small synthetic training set does not establish domain generalization. The experiment is preserved under `experiments/`.

**Reproducibility caveat:** The checked-in checkpoint directory does not contain a weights file; experiment evaluation from a fresh clone requires regenerating or restoring weights. Do not characterize that experiment as a validated production model.

### 4.4 Qwen-only filtering

Provides semantic relationship labels and produced perfect top-three results on the six hand-authored binary scenarios. Because Qwen emits labels rather than continuous relevance scores, the Qwen-only benchmark preserved the input candidate order.

**Why E5 was still retained:** It offers a deterministic semantic ordering mechanism for situations with more accepted candidates than display positions. This is a design rationale, **not measured proof that reranking helps**; further graded evaluation is needed.

## 5. Evaluation methodology and observed results

Evaluation sources:

- `data/evaluation/relationship_dev.json`: 20 hand-authored **development** pairs used for classifier/prompt development.
- `data/evaluation/relationship_test.json`: 100 proposed synthetic labeled cases evaluated after the classifier prompt was frozen.
- `data/evaluation/*_predictions.csv`: recorded case-level predictions.
- `evaluation/evaluate_ranking.py` and `data/evaluation/ranking_results.csv`: six small cart scenarios with three binary-labeled relevant candidates each and five candidate items per scenario.
- `tests/test_recommendations_api.py`: mocked backend contract/business-rule tests.

### Classification (recorded local results, not re-executed for this document)

| Measure | Development (20 pairs) | Test (100 pairs) |
|---|---:|---:|
| Accuracy | 90% (18/20) | **81% (81/100)** |
| Complement precision | 100% on 12 dev complement examples* | **79.6% (39/49)** |
| Complement recall | 100% on 12 dev complement examples* | **97.5% (39/40)** |
| Complement F1 | Development results are not independently representative | **87.6%** |
| Substitute recall | 50% (2/4) | **50%** |

*Development figures are partly from prompt-tuning work and should not be presented as held-out validation. In particular, complement precision on development pairs depends on all predicted-positive cases; the main takeaway is 12/12 reference complements identified.

Test macro-F1 for all four labels was approximately **69.5%** (three substantive labels approximately **82.4%**). The four labels were `complement`, `substitute`, `unrelated`, and `uncertain`. The benchmark's label source is `synthetic_proposed`; the cases are **not verified production MyThings ground truth**.

**Important failure modes:** fries versus curly fries, closely related sandwich variants, and unexpected pairings involving sweet sauces. These show why high complement recall alone is insufficient; false positives risk visibly inappropriate add-ons. The reported classifier's substitute recall of 50% is an important weakness.

### Ranking/ablation (six manually constructed scenarios)

| Pipeline | Mean Precision@3 | Mean NDCG@3 | False top-three recommendations |
|---|---:|---:|---:|
| E5 only | **0.556** | **0.500** | **8** |
| Qwen only | **1.000** | **1.000** | **0** |
| Qwen + E5 | **1.000** | **1.000** | **0** |

These are **local, limited ablation results**. Every scenario provides exactly **three** binary relevant items and asks for three recommendations. The Qwen filter accepted exactly those labeled items in these scenarios. Therefore, P@3 and binary NDCG@3 **cannot distinguish** a changed ordering of those accepted items: this experiment supports the **benefit of filtering compared with standalone E5**, but **does not establish that reranking outperforms Qwen-only**.

Latency from the recorded files varied by scenario and was influenced by model warm-up/caching. These measurements are not production P95 or reliable end-to-end capacity estimates. Re-running a test with changed model code, cache state, or hardware requires a new recorded result.

### Testing scope

The five previously passing API unit tests monkeypatch the Qwen filter and E5 ranker. They verify request validation, merchant isolation, exclusion of cart items, empty-cart responses, and top-K logic; **they do not validate real-model inference**. A successful Docker build, C# compilation, startup health check, and live recommendation request must be verified independently.

## 6. Why not train a recommendation model from scratch?

A learned purchase or complementarity model would ideally need:

- Historical item-to-item co-purchases and cart/session sequences;
- Catalog attributes, merchant IDs, product availability, and prices;
- Reliable positive/negative labels (with careful handling of exposure bias);
- A time-based and merchant-disjoint evaluation split;
- Offline ranking metrics plus online A/B testing of attachment rate, conversion, average order value, and customer satisfaction.

Those inputs were **not available** in this prototype. Consequently, a pretrained multilingual language model plus pretrained embeddings was selected to demonstrate a feasible product-behavior prototype without making unsupported claims about personalized ranking or conversion uplift. Generating larger synthetic datasets alone would not substitute for real behavioral validation.

## 7. Planned improvements and alternatives requiring more time

| Priority | Future approach | Why it matters | Required work |
|---|---|---|---|
| High | Human-reviewed Arabic/Jordanian benchmark | Improve evaluation validity and address dialect ambiguity | Annotation guidelines, multiple reviewers, conflict resolution, domain/merchant stratification |
| High | Merchant-authorized candidate shortlist | Avoid running pairwise Qwen on every catalog item | Search/embedding index or SQL filters; merchant and availability gating |
| High | Real-model end-to-end tests and readiness checks | Separate service liveness from successful inference | Integration fixtures, model version control, startup/model checks |
| High | Failure-cache policy | Current cache can indefinitely preserve transient model failures | Only cache successful decisions, or use TTL and retry/backoff |
| High | Operational safeguards | API currently trusts upstream metadata, lacks authorization and rate limits | Trusted internal network, auth, request limits, logging/metrics, timeouts, circuit breaker |
| Medium | Graded relevance and >3 viable complements | Test whether E5 reranking actually improves ordering | Blind human judgments, larger candidate sets, order-sensitive NDCG/MRR |
| Medium | Product compatibility constraints | Generic semantics cannot verify USB-C generation, voltage, size, allergies | Structured attributes, deterministic compatibility rules or trusted metadata |
| Medium | Better multilingual reranking / domain-tuned cross-encoder | Improve fine-grained rank quality | Collected domain labels, training/validation, comparisons against simpler baselines |
| Medium | Async/batched/parallel model pipeline | Improve latency as cart and candidates grow | Concurrency bounds, profiling, queueing, GPU service management |
| Medium | Personalized/co-purchase recommendation systems | Adapt to actual shopping preferences and repeated behavior | Privacy-conscious event logging, training data, counterfactual evaluation |
| Later | Production CI/CD, GPU containers, scalable inference | Reliable operations at larger throughput | Dependency pinning, integration tests, image scanning, GPU scheduling, observability |
| Later | Online A/B test | Determine whether recommendations create business value | Product analytics, treatment/control assignment, business guardrails |

### Suggested production evolution

```text
Authorized .NET commerce API + inventory/merchant rules
                |
       Small eligible shortlist
                |
  Fast multilingual retrieval / co-purchase candidates
                |
     Pairwise relationship verification
                |
    Learned or domain-validated reranking
                |
    Business constraints and diversification
                |
    Measured online results + human feedback
```

This is a proposed evolution, **not functionality already shipped**.

## 8. Deployment and integration assumptions

- Local: Python + Ollama installed separately.
- Docker Compose: API and Ollama run in separate containers; `docker compose exec ollama ollama pull qwen2.5:7b` downloads model weights into a persistent volume. Docker avoids a separate Ollama host installation but **does not avoid hardware and model-download requirements**.
- The API reads `OLLAMA_URL` from an environment variable when set. Local default: `http://127.0.0.1:11434/api/chat`. Within Compose: `http://ollama:11434/api/chat`.
- The C# example demonstrates JSON DTOs and an `HttpClient` call. The commerce system must supply trusted product data and revalidate returned IDs.
- This repository does **not** include live company credentials, production integration, Docker GPU settings, or security hardening.

See [Docker instructions](DOCKER.md), [C# integration example](../integration/dotnet/README.md), and the [root README](../README.md).

## 9. Bottom line

The available evidence supports a **working conceptual architecture and promising filtering behavior on small synthetic cases**, not a claim of production recommendation quality. The engineering contribution is the explicit separation of relationship verification from semantic ranking, the bilingual metadata pipeline, reproducible evaluation artifacts, API boundary, and transparent statement of limitations. The most important next step for meaningful real-world value is **representative merchant/catalog data and rigorous review of false-positive complements**, followed by latency and online product metrics.

## 10. Latency reduction strategy (proposed; not yet benchmarked)

### 10.1 Identify the bottleneck before changing the pipeline

The current service sends **one synchronous Qwen/Ollama request for each cart-product/candidate pair**. With 2 items in the cart and 20 eligible candidates, that can mean 40 classification calls before E5 ranks accepted candidates. Response time therefore grows approximately with the number of evaluated pairs, subject to caching, model warm-up, and actual inference conditions.

Measure separately: eligibility filtering, Qwen cold vs warm inference, per-pair median/P95 latency, number of calls, cache-hit rate, E5 model initialization and encoding, and full HTTP response time. Compare equal hardware and model versions, recording whether caches were cleared. These are **future measurements**, not values inferred from the small synthetic benchmark.

### 10.2 Candidate optimizations and tradeoffs

| Optimization | Potential latency benefit | Risk / necessary validation | Suggested priority |
|---|---|---|---|
| **Keep Ollama/Qwen loaded and use GPU acceleration** | Reduce model loading and token-generation time | GPU memory limits; current Compose configuration does not request GPU access; CPU-only may exceed the 30-second pair timeout | High |
| **Reuse successful cached pair judgments** | Avoid redundant calls for recurring product pairs | Current process-local `lru_cache` also stores `uncertain` when Ollama fails; distinguish actual uncertainty from transient failures; consider bounded TTL or Redis only if needed | High |
| **Reduce candidate count with trusted business constraints** | Fewer Qwen comparisons without introducing approximate semantic filtering | Must not accidentally remove legitimate options; merchant authorization, stock, cart exclusions, and attribute compatibility belong to the commerce layer | High |
| **Control simultaneous Qwen requests** | May improve wall-clock time for independent comparisons | Too much concurrency can saturate one GPU, queue requests, and increase P95; compare 1, 2, and 4 workers under load | Medium |
| **Classify multiple candidates in one structured LLM response** | Lower repeated prompt/HTTP overhead | Larger prompts, truncated output, missed candidate IDs, cross-item contamination; validate parsing and accuracy | Medium / experimental |
| **Precompute/cache E5 product embeddings** | Reduce repeated encoding for largely static catalogs | Need product versioning, invalidation, and consistent model version; does not address Qwen as dominant bottleneck | Medium |
| **Evaluate a smaller/quantized language model** | Potentially faster inference and smaller hardware footprint | Possible degradation in Arabic understanding, substitute recall, and complement precision; compare on a genuinely held-out, human-reviewed test set | Medium / experimental |
| **E5 preliminary shortlist before Qwen** | Fewer expensive Qwen calls on large catalogs | **Candidate-recall failure:** E5 may drop true complements before Qwen sees them; not recommended as an unvalidated default | Research only |

### 10.3 Why an E5-first shortlist is risky

The saved six-scenario ranking results show that **E5 alone** placed **Chicken Burger** and **Chocolate Dessert Sauce** above useful accompaniments in the *Beef Burger* scenario. A top-three E5 shortlist would pass those two unsuitable candidates and **exclude Garlic Sauce and French Fries**. Qwen could reject the bad candidates, but it could **not recover products already discarded by E5**.

This is a retrieval **recall** problem, not simply a final ranking error. A shortlist of ten or twenty might help at larger catalog sizes, but we have **not measured** its retained-complement recall. The current six scenarios each have only five candidates, so they cannot establish an appropriate shortlist size.

**Current recommendation:** Keep **Qwen filtering before E5 ranking** in the evaluated submission pipeline. Explore E5 preselection only in a separate experiment using larger, independently labeled candidate pools. Measure `Recall@M` of known complementary candidates for several shortlist sizes alongside end-to-end latency and final Precision@3/NDCG@3. Do not promote an optimization that causes material recall loss.

### 10.4 Proposed implementation and test sequence

1. Establish cold-start and warm-cache latency baselines for carts with 1, 2, and 3 items and candidate pools of 5, 20, and 100 products. Record Qwen calls, median/P95, total response time, and hardware configuration.
2. Fix transient-failure caching without changing semantic labels: cache only validated model responses; log timeouts separately from genuine `uncertain` predictions.
3. Confirm whether Ollama runs on a compatible GPU; test persistent model loading and the existing Compose configuration on the target machine. Avoid assuming Docker automatically enables GPU acceleration.
4. Test modest, bounded concurrency and optionally a batched-classification prototype. Re-evaluate classification quality as well as request time; do not assume parallelism guarantees higher throughput.
5. Cache product embeddings when catalog size justifies it, with explicit invalidation on metadata/model changes.
6. Only then compare alternate models or first-stage retrieval approaches under **equal recall/quality constraints**.

### 10.5 Quality and operational guardrails

Latency optimization must not hide regressions. Track **complement precision and recall**, **substitute recall** (50% in the reported 100-case benchmark), inappropriate recommendation count, ranking metrics when graded relevance is available, timeout/error rate, model cache hit rate, memory/VRAM, and response-time percentiles. Make a distinction between per-pair inference latency, cached latency, and true end-to-end cart latency.

No latency reduction figures for these proposed optimizations are claimed here. Their impact depends on hardware, candidate volume, caching, and the chosen model. Optimizing the Qwen bottleneck is promising, but deployment decisions require a new benchmark and no unverified loss of useful recommendations.

