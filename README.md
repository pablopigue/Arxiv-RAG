# arXiv RAG - ask machine-learning papers, get cited answers

A retrieval-augmented generation system that answers questions over a corpus of
arXiv papers and **grounds every answer in cited sources**. Runs fully locally and
free with [Ollama](https://ollama.com) — no API keys, no per-token costs.

Built to demonstrate a production-shaped RAG pipeline end to end: ingestion,
chunking, dense retrieval with cross-encoder reranking, grounded generation,
an evaluation harness, a REST API, and a UI.

---

## Built for Production

Most RAG demos stop at "embed a PDF and ask a question." This one adds the parts that
matter in practice:

- **Two-stage retrieval** — fast bi-encoder recall, then a cross-encoder reranker. The
  eval harness *quantifies the reranker's lift* so the design choice is measured, not assumed.
- **Grounded generation with citations** — the model must answer only from retrieved
  context and cite sources inline `[1][2]`, or explicitly abstain.
- **Evaluation built in** — Hit@k and MRR for retrieval, plus an LLM-as-judge
  faithfulness score for generation.
- **Swappable LLM backend** — the pipeline depends on an `LLM` protocol, so moving from
  Ollama to Claude/OpenAI is one class and one line.
- **Servable** — FastAPI endpoint + Streamlit UI + Docker Compose (app + Ollama).

---

## Architecture

```mermaid
flowchart LR
    A[arXiv API] -->|PDFs| B[PyMuPDF parser]
    B --> C[Recursive chunker<br/>+ metadata]
    C --> D[bge embeddings]
    D --> E[(ChromaDB)]

    Q[User question] --> F[Dense retrieval<br/>top_k=20]
    E --> F
    F --> G[Cross-encoder<br/>rerank → top_n=5]
    G --> H[Grounded prompt<br/>with numbered sources]
    H --> I[Ollama LLM]
    I --> J[Answer + citations]
```

**Stack:** Python · sentence-transformers (`bge-small-en-v1.5`) · cross-encoder reranker
· ChromaDB · Ollama · FastAPI · Streamlit · Docker

---

## Quickstart

> Run every command from the project root (the folder with this README). The entry-point
> scripts add the root to Python's path automatically, so `python scripts/...` just works.

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Start a local LLM
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1
```

### 3. Build the index
```bash
python scripts/build_index.py \
  --query "cat:cs.LG AND abs:retrieval augmented generation" --max 30
```

### 4. Ask
```bash
python scripts/ask.py "What is the difference between a bi-encoder and a cross-encoder?"
```

Or launch the UI / API:
```bash
streamlit run src/app/streamlit_app.py      # http://localhost:8501
uvicorn src.api.main:app --reload           # http://localhost:8000/docs
```

### Docker (app + Ollama together)
```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1
```

---

## Evaluation

```bash
python eval/eval_dataset.py          # bootstrap a synthetic QA set from the corpus
python eval/evaluate.py --faithfulness
```

Example output and analysis:

| Metric                | Dense only | + Reranker |
|-----------------------|:----------:|:----------:|
| Hit@5                 |    0.44    |    0.54    |
| MRR                   |    0.35    |    0.44    |
| Faithfulness (judge)  |    0.73    |    0.72    |
| Abstentions (of 50)   |     2      |     0      |

> The synthetic set is bootstrapped: the LLM writes a question answerable only from a
> sampled chunk, and that chunk becomes the gold passage.
> Reranking clearly improves retrieval (MRR +0.09, Hit@5 +0.10) and removes
> abstentions (2 -> 0): better passages let the model answer questions where dense
> retrieval came up short. Faithfulness stays flat (~0.72) - it measures whether the
> answer sticks to its sources, which the reranker doesn't change.

---

## Project layout

```
src/
  config.py              # all tunables in one place (pydantic-settings)
  ingest/                # arXiv loader · PDF parsing · chunking
  retrieval/             # embeddings · ChromaDB store · reranking retriever
  generation/            # swappable LLM interface · RAG pipeline
  api/main.py            # FastAPI service
  app/streamlit_app.py   # demo UI
scripts/                 # build_index.py · ask.py
eval/                    # synthetic QA set + metrics harness
tests/                   # unit tests
```

---
