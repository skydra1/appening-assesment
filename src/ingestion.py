import argparse
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME,
    PDF_PATH,
    get_embeddings,
    validate_keys,
)


def ensure_pinecone_index(index_name: str, dimension: int = EMBEDDING_DIMENSION) -> None:
    """Create the Pinecone serverless index if it does not exist."""
    validate_keys(require_pinecone=True)
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name in [i.name for i in pc.list_indexes()]:
        print(f"Pinecone index '{index_name}' already exists.")
        return
    print(f"Creating Pinecone index '{index_name}' (dim={dimension}, cosine)...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until ready
    for _ in range(30):
        if pc.describe_index(index_name).status.ready:
            break
        time.sleep(2)
    print(f"Index '{index_name}' is ready.")


def load_and_chunk(pdf_path: str):
    """Load a PDF and split it into chunks. Returns (docs, chunks)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'"
        )
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {pdf_path}.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return docs, chunks


def run_ingestion(pdf_path: str = PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """Full ETL: load -> chunk -> embed -> upsert into Pinecone."""
    validate_keys(require_pinecone=True)
    _, chunks = load_and_chunk(pdf_path)
    ensure_pinecone_index(index_name)
    embeddings = get_embeddings()
    print(f"Upserting {len(chunks)} chunks into index '{index_name}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
    print("Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to source PDF")
    parser.add_argument("--index", default=PINECONE_INDEX_NAME, help="Pinecone index name")
    args = parser.parse_args()
    run_ingestion(pdf_path=args.pdf, index_name=args.index)
