from typing import List, Dict, Any
from agents.rag_agents.indexing_agent import get_retriever_with_metadata

def retrieve_with_metadata(query: str, k: int = 5) -> List[Dict[str, Any]]:
    retriever = get_retriever_with_metadata(k=k)
    docs = retriever.invoke(query)
    
    results = []
    for d in docs:
        results.append({
            "page_content": d.page_content,
            "metadata": dict(d.metadata or {}),
        })
    return results

__all__ = ["retrieve_with_metadata"]