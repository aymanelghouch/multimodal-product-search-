# Multimodal Product Search

Visual and semantic search engine for a fashion catalog. Find products by **text**, by **image**, or both, using CLIP embeddings and (soon) a vector database.

> 🚧 **Work in progress.** Phases 1–3 are done. Next: reproducible embedding pipeline, Qdrant, evaluation, FastAPI and demo.

## Results so far

Every method is evaluated on 1,000 products (25 article types × 40). For each product used as a query, Precision@5 is the fraction of the 5 nearest results with the same `articleType`.

| Method                                           | Precision@5 |
| ------------------------------------------------ | ----------- |
| Random baseline                                  | 0.039       |
| Color histogram (8 bins per channel) + cosine    | 0.226       |
| Color histogram without the white-background bin | 0.165       |
| **CLIP image embeddings + cosine**               | **0.781**   |

CLIP is about 3.5× better than the color-histogram baseline, even though catalog images are only 60×80 px.

Text search works too. For example, `silver wrist watch` returns 5 watches, and `black running shoes` returns black shoes (some of them formal, not running).

**Caveats:** the metric is strict (a casual shoe returned for a sports-shoe query counts as an error) and may be inflated by near-duplicate products (same item in different colors). Text search has only been checked qualitatively so far.

## Roadmap

- [x] **Phase 1:** Data preparation (1,000 stratified products)
- [x] **Phase 2:** Baseline visual search with NumPy (color histograms, cosine similarity, Top-K)
- [x] **Phase 3:** CLIP embeddings (image and text) and text search
- [ ] **Phase 4:** Embedding pipeline (batching, caching, resumable)
- [ ] **Phase 5:** Qdrant vector database
- [ ] **Phase 6:** Multimodal and hybrid search
- [ ] **Phase 7:** Evaluation (Recall@K, MRR, NDCG, latency)
- [ ] **Phase 8:** FastAPI service
- [ ] **Phase 9:** Demo, tests, docs

## Project structure

```
multimodal_search_engine/
├── scripts/
│   ├── prepare_dataset.py     # select and save 1,000 products
│   ├── phase2_baseline.py     # color-histogram baseline
│   └── phase3_clip.py         # CLIP embeddings, evaluation and text search
├── src/
│   ├── features.py            # color histograms
│   ├── similarity.py          # L2 normalization, cosine Top-K
│   └── evaluation.py          # Precision@K
├── tests/
├── data/                      # not tracked (see below)
├── NOTES.md                   # learning log with measurements and mistakes
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Generate the data (not tracked in git) and run each phase from the project root:

```bash
python -m scripts.prepare_dataset
python -m scripts.phase2_baseline
python -m scripts.phase3_clip
```

`prepare_dataset` downloads the catalog from Hugging Face and writes `data/images/` and `data/processed/products.csv`. `phase3_clip` computes the CLIP embeddings and saves them to `data/processed/`.

## Dataset

Fashion product catalog from Hugging Face (44,072 items, 60×80 px images). This project uses a stratified sample of 1,000 products so each of the 25 most frequent article types is equally represented.

## Design decisions

- **Stratified sample** by `articleType`: a random sample would be about half `Apparel` and almost empty for other categories, which makes evaluation meaningless.
- **Reproducible selection** with a fixed seed.
- **Embeddings saved unnormalized**, so normalization can change without recomputing them.
- **Evaluation code is independent of the representation**: the same `precision_at_k` evaluates histograms and CLIP.

## Known limitations

- Low-resolution images (60×80) are upscaled to 224×224 for CLIP.
- About 30% of product names in the full catalog are duplicated, which can inflate retrieval metrics.
- Relevance is defined by `articleType` only; it does not account for color or style.
