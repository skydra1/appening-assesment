# RAG-based AI Chatbot — Agentic AI eBook

A Retrieval-Augmented Generation chatbot that answers questions strictly from the
**Agentic AI eBook**, built with **LangGraph + Pinecone** and exposed via **FastAPI**
(or a **Streamlit** UI).

## Architecture

```text
PDF (data/Ebook-Agentic-AI.pdf)
  -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000/200)
  -> embeddings (OpenAI-compatible API, dim 1536)
  -> Pinecone index (cosine)

Query -> LangGraph [ retrieve (top-k=3) -> generate (strict prompt, temp=0) ]
  -> { final_answer, retrieved_context, confidence_score }
```

- `src/config.py` — env vars and shared constants.
- `src/ingestion.py` — PDF loading, chunking, Pinecone index setup, upsert. CLI included.
- `src/graph.py` — `AgentState`, retrieve + generate nodes, compiled graph.
- `app.py` — FastAPI `POST /chat` (+ `/health`).
- `streamlit_app.py` — chat UI with a side panel for context chunks and score.
- `tests_sample_queries.py` — 6 benchmark queries (last one checks the model refuses
  out-of-context questions).

## Setup (Windows PowerShell)

```powershell
# 1. Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Configure keys
Copy-Item .env.example .env
# edit .env: your API key, base URL, Pinecone key + index name

# 4. Add the eBook PDF
# Download https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing
# Save as data\Ebook-Agentic-AI.pdf

# 5. Ingest (chunk + embed + upsert to Pinecone, creates the index if missing)
python -m src.ingestion

# 6a. Run FastAPI
python -m uvicorn app:app --reload
# POST http://127.0.0.1:8000/chat  body: {"query": "What is Agentic AI according to the eBook?"}

# 6b. Or run Streamlit
streamlit run streamlit_app.py

# 7. Run benchmark queries
python tests_sample_queries.py          # needs the server running
python tests_sample_queries.py --direct # calls the graph directly, no server needed
# full output is also saved to test_results.json
```

## Create the Pinecone vector DB

The index needs dimension **1536** (matches the default embedding model) and metric
**cosine**. Easiest path: just run ingestion, it creates the index if missing
(`ensure_pinecone_index` in `src/ingestion.py`):

```powershell
python -m src.ingestion
```

## Any OpenAI-compatible model works

All chat and embedding calls go through `OPENAI_BASE_URL` (defaults to
`https://api.openai.com/v1`). To use another provider, change the base URL and the
model names in `.env` — no code changes needed. If you switch to an embedding model
with a different vector size, create a matching Pinecone index since dimensions are
fixed at creation time.

## API contract

Request:

```json
{ "query": "What role does memory play in Agentic AI workflows?" }
```

Response:

```json
{
  "final_answer": "...",
  "retrieved_context": ["chunk1", "chunk2", "chunk3"],
  "confidence_score": 0.95
}
```

Grounding rule: if the retrieved chunks don't contain the answer, the model replies
exactly `I cannot answer based on the provided document.` with score `0.0`
(e.g. *"Who won the 2022 FIFA World Cup?"*).

## Confidence score

Coarse heuristic in `src/graph.py:compute_confidence` — `0.0` when there is no
context or the model refused, otherwise it grows slightly with the number of
retrieved chunks (capped at `0.95`). So a normal grounded answer scores `0.95`
and anything ungrounded scores `0.0`. It reflects whether grounding happened, not
a measured relevance.
