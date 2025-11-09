"""
Retrieval Agent: Retrieves relevant chunks from vector DB for RAG.
"""
from typing import List, Dict, Any
import re
from agents.rag_agents.indexing_agent import get_retriever_with_metadata

def retrieve_with_metadata(query: str, k: int = 5) -> List[Dict[str, Any]]:
  """Retrieve documents with full metadata support."""
  retriever = get_retriever_with_metadata(k=k)
  docs = retriever.invoke(query)  # Returns Document objects with metadata
  
  results: List[Dict[str, Any]] = []
  for d in docs:
      results.append({
          "page_content": d.page_content,
          "metadata": dict(d.metadata or {}),
      })
  return results

__all__ = ["retrieve_with_metadata"]