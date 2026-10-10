# Multimodal Product Search

Visual and semantic product search engine for a fashion catalog:
search by text, by image, or both, using CLIP embeddings and a vector database.

> 🚧 Work in progress. Currently: Phase 1 (data preparation) done.

## Roadmap

- [x] Phase 1: Data preparation (1,000 stratified products)
- [ ] Phase 2: Baseline visual search with NumPy (cosine similarity)
- [ ] Phase 3: CLIP embeddings
- [ ] Phase 4: Embedding pipeline
- [ ] Phase 5: Qdrant
- [ ] Phase 6: Multimodal and hybrid search
- [ ] Phase 7: Evaluation (Recall@K, MRR, NDCG, latency)
- [ ] Phase 8: FastAPI
- [ ] Phase 9: Demo, tests, docs

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_dataset.py
```

## Dataset

Fashion product images catalog from Hugging Face (44k items).
This project uses a stratified sample of 1,000 products (25 article types × 40).
