"""FastAPI backend exposing the RAG graph at POST /chat."""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PINECONE_INDEX_NAME
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        try:
            _graph = build_rag_graph(
                index_name=os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _graph


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question about the eBook")


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {"status": "ok", "service": "Agentic AI RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    graph = get_graph()
    initial_state = {"question": request.query, "context": [], "answer": "", "score": 0.0}
    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG pipeline failed: {exc}") from exc
    return QueryResponse(
        final_answer=result.get("answer", ""),
        retrieved_context=result.get("context", []),
        confidence_score=float(result.get("score", 0.0)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
