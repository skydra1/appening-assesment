import os
import time

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION: int = 1536 
LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
RETRIEVER_TOP_K: int = 3

PDF_PATH: str = os.path.join("data", "Ebook-Agentic-AI.pdf")

# Canonical refusal.
REFUSAL_PHRASE = "I cannot answer based on the provided document."


def validate_keys(require_pinecone: bool = True) -> None:
    """Raise a clear error if required API keys are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if require_pinecone and not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )


class StringInputEmbeddings(OpenAIEmbeddings):
    # Stock OpenAIEmbeddings sends token-id arrays, which only OpenAI
    # accepts. Other providers want plain strings, so skip tokenizing
    # and send each text as-is (chunks are ~1000 chars, well within limits).
    def _tokenize(self, texts, chunk_size):  # type: ignore[override]
        counts = [max(1, len(t) // 4) for t in texts]
        step = chunk_size or len(texts) or 1
        return range(0, len(texts), step), list(texts), list(range(len(texts))), counts

    request_batch_size: int = 20  # texts per API call (provider batch caps)
    throttle_seconds: float = 4.0  # pause between calls (free-tier quotas)

    def embed_documents(self, texts, chunk_size=None):
        # NOTE: super() still sends raw strings thanks to _tokenize above;
        # this wrapper only splits the work into small, spaced-out calls.
        out = []
        step = chunk_size or self.request_batch_size
        for i in range(0, len(texts), step):
            if i > 0:
                time.sleep(self.throttle_seconds)
            out.extend(super().embed_documents(texts[i : i + step]))
        return out


def get_embeddings() -> StringInputEmbeddings:
    validate_keys(require_pinecone=False)
    return StringInputEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        dimensions=EMBEDDING_DIMENSION,
        chunk_size=50,
    )