from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PINECONE_INDEX_NAME,
    REFUSAL_PHRASE,
    RETRIEVER_TOP_K,
    get_embeddings,
    validate_keys,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


SYSTEM_PROMPT = (
    "You are a strict assistant for the 'Agentic AI' eBook. "
    "Answer the question relying ONLY on the context below. "
    "Do not use any outside knowledge. "
    f"If the context does not contain enough information, reply exactly: '{REFUSAL_PHRASE}'"
)


def compute_confidence(context: List[str], answer: str) -> float:
    # 0 when no grounding is possible, scales with chunk count.
    if not context:
        return 0.0
    if REFUSAL_PHRASE.lower() in (answer or "").lower():
        return 0.0
    return min(0.95, 0.65 + 0.10 * len(context))


def build_rag_graph(index_name: str = PINECONE_INDEX_NAME, top_k: int = RETRIEVER_TOP_K):
    validate_keys(require_pinecone=True)

    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
    )

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state.get("context", []))
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
        response = llm.invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        score = compute_confidence(state.get("context", []), answer)
        return {"answer": answer, "score": score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def answer_question(question: str, index_name: str = PINECONE_INDEX_NAME) -> dict:
    graph = build_rag_graph(index_name=index_name)
    initial_state: AgentState = {"question": question, "context": [], "answer": "", "score": 0.0}
    return graph.invoke(initial_state)
