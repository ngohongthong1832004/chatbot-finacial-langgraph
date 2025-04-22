from typing import Dict, Any
from .init_vector_db import retrieve
from langgraph.graph import StateGraph, END
from chatfinan.database.connect import query_sql
from .tool import web_search, decide_to_generate
from .documents import grade_documents, rewrite_query, generate_answer
from .graph_model import GraphState

# Initialize graph
def create_rag_graph():
    agentic_rag = StateGraph(GraphState)
    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("rewrite_query", rewrite_query)
    agentic_rag.add_node("web_search", web_search)
    agentic_rag.add_node("query_sql", query_sql)
    agentic_rag.add_node("generate_answer", generate_answer)
    # ✅ Entry point
    agentic_rag.set_entry_point("retrieve")
    # Edges
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
            "query_sql": "query_sql"
        }
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("query_sql", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)
    return agentic_rag.compile()

def process_query(query: str) -> Dict[str, Any]:
    """
    call pipeline Agentic RAG process
    """
    print(f"📥 Processing query: {query}")
    rag_graph = create_rag_graph()
    result = rag_graph.invoke({"question": query})
    print(f"📤 Done. Generation: {result.get('generation', '')[:100]}")
    return result
