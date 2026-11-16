# Experimental approaches

This folder preserves earlier approaches considered before selecting the final **Qwen2.5 7B relationship filter + multilingual E5 reranker**.

**These modules are not called by the production recommendation endpoint.** In particular, moving experimental scoring here does not alter the reported classification or ranking results.

## What is here

| File | Purpose |
|---|---|
| `baseline.py` | Original handwritten category compatibility baseline |
| `complementarity.py` / `experiment_pairing.py` | Earlier E5 query-to-product category-pairing heuristic |
| `experiment_contrastive.py` | Positive-vs-negative E5 prompt comparisons |
| `cross_encoder/pairwise_ranker.py` / `experiment_cross_encoder.py` | Pretrained multilingual cross-encoder ranking experiment |
| `create_pair_dataset.py` | Generate small synthetic pairwise training/evaluation sets |
| `train_pairwise.py` | Historical cross-encoder fine-tuning script |
| `evaluate_pairwise.py` | Evaluate the saved experimental cross-encoder model |
| `evaluate_model.py` | Small E5-only ranking sanity-check scenarios |
| `test_embeddings_manual.py` | Inspect embedding dimensions |
| `test_relationships_manual.py` | Manually inspect current Qwen relationship labels |

## Commands

Run from the **repository root** with a Python environment installed using `uv sync`:

```powershell
uv run python -m experiments.experiment_pairing
uv run python -m experiments.experiment_contrastive
uv run python -m experiments.experiment_cross_encoder
uv run python -m experiments.test_embeddings_manual
uv run python -m experiments.test_relationships_manual
uv run python -m experiments.create_pair_dataset
```

The Qwen classifier experiments require a running Ollama instance with `qwen2.5:7b`; E5 and cross-encoder experiments download their pretrained models on first use.

The historical fine-tuning workflow reads `data/training/train_pairs.json` and evaluates against `data/training/eval_pairs.json`:

```powershell
uv run python -m experiments.train_pairwise
uv run python -m experiments.evaluate_pairwise
```

**Checkpoint caveat:** The repository's `artifacts/pairwise_complementarity_v1/` contains tokenizer/configuration files but does **not** include the model weight file, which is ignored by `.gitignore`. Therefore `evaluate_pairwise.py` cannot load this checkpoint from a fresh clone unless you regenerate or restore its model weights. The cross-encoder model is experimental and **not** the deployed Qwen/E5 pipeline.

## Evaluation integrity

- Generated training examples and holdout pairs are tiny, synthetic examples rather than observed purchase behavior.
- Some scripts use illustrative prompts and handcrafted relevance labels.
- Results from experiments should not be presented as accuracy on actual MyThings orders or as evidence of production conversion lift.
- Keep the final Qwen classifier prompt frozen when comparing against the recorded test set.

For the final service's evaluation procedure and results, see the [root README](../README.md) and `evaluation/`.
