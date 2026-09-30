import os

import streamlit as st

from src.config import PINECONE_INDEX_NAME
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")
st.title("Agentic AI RAG Chatbot")
st.caption("Strictly grounded answers from the Agentic AI eBook (Pinecone + LangGraph).")


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_graph():
    return build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME))


try:
    graph = load_graph()
except RuntimeError as exc:
    st.error(str(exc))
    st.info("Copy `.env.example` to `.env` and add your OPENAI_API_KEY / PINECONE_API_KEY, then re-run.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# each turn shows the question then its answer.
for turn in st.session_state.messages:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(f"Confidence: {turn['score']:.2f} | {len(turn['context'])} chunk(s)")

question = st.chat_input("Ask a question about the Agentic AI eBook...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Retrieving context and generating answer..."):
        result = graph.invoke({"question": question, "context": [], "answer": "", "score": 0.0})
    turn = {
        "question": question,
        "answer": result.get("answer", ""),
        "score": float(result.get("score", 0.0)),
        "context": result.get("context", []),
    }
    st.session_state.messages.append(turn)
    st.session_state.last_result = turn
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(f"Confidence: {turn['score']:.2f} | {len(turn['context'])} chunk(s)")

with st.sidebar:
    st.header("Retrieved context")
    last = st.session_state.last_result
    if last is None:
        st.write("Ask a question to see the retrieved chunks and score here.")
    else:
        st.write(f"Latest question: {last['question']}")
        st.metric("Confidence score", f"{last['score']:.2f}")
        st.write(f"{len(last['context'])} chunk(s)")
        for i, chunk in enumerate(last["context"], start=1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)
