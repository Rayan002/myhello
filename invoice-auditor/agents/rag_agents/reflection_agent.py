"""
Reflection Agent: Reflects on RAG retrieval results and improves query/answer quality.
"""
import os
from typing import Dict, List, Any

from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

# Load .env variables (like AWS credentials) at the start
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
  
# LLM for RAGAS evaluation (requires 1 temperature for consistency)
evaluator_llm_instance: BaseChatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def reflect_on_answer(answer: str, retrieved_docs: str, question: str = "") -> str:
 """
 Reflect on retrieval quality and suggest improvements.
 
 Args:
     answer: Generated answer
     retrieved_docs: Retrieved documents with metadata
     question: Original question
     
 Returns:
     "good" or "bad"
 """
 
 prompt = ChatPromptTemplate.from_messages([
     ("system", """You are given with the docs retrieved for the user query and the answer made by LLM.
         Your task is to analyze the generated answer that it is aligned with the retrieved docs
         and give a single word response telling the good, bad.
     """),
     ("human", "Retrieved Docs:\n{doc_summary} and Answer: {answer}\n\n"),
 ])
 
 chain = prompt | llm
 response = chain.invoke({"doc_summary" : retrieved_docs, "answer" : answer})
 return response.content.strip().lower()


def calculate_ragas_metrics(question: str, answer: str, contexts: List[str], docs: List[Dict[str, Any]] = None) -> Dict[str, float]:
 """
 Calculate metrics using LangChain evaluators.
 Simple and straightforward implementation.
 
 Args:
     question: User's question
     answer: Generated answer
     contexts: List of context strings
     docs: Optional list of document dicts (for compatibility with chatbot_workflow)
 
 Returns:
     Dictionary with metrics: groundedness, relevance, context_relevance
 """
#  from langchain.evaluation import load_evaluator
 
#  # Combine contexts into single string
#  context_text = "\n".join(contexts) if contexts else ""
 
#  scores = {
#      "groundedness": 0.0,
#      "relevance": 0.0,
#      "context_relevance": 0.0  # Maps to context_utilization for compatibility
#  }
 
#  # 1. Groundedness: Check if answer is supported by contexts
#  try:
#      groundedness_eval = load_evaluator(
#          "labeled_criteria",
#          criteria="correctness",
#          llm=evaluator_llm_instance
#      )
#      result = groundedness_eval.evaluate_strings(
#          prediction=answer,
#          input=question,
#          reference=context_text
#      )
#      scores["groundedness"] = float(result.get("score", 0.0))
#  except Exception as e:
#      print(f"Groundedness evaluation failed: {e}")
 
#  # 2. Relevance: Check if answer addresses the question
#  try:
#      relevance_eval = load_evaluator(
#          "labeled_criteria",
#          criteria="relevance",
#          llm=evaluator_llm_instance
#      )
#      result = relevance_eval.evaluate_strings(
#          prediction=answer,
#          input=question,
#          reference=context_text  # Add this
#      )
#      scores["relevance"] = float(result.get("score", 0.0))
#  except Exception as e:
#      print(f"Relevance evaluation failed: {e}")
 
#  # 3. Context Utilization: Simple word """ """  """ """overlap calculation
#  try:
#      if context_text and answer:
#          answer_words = set(answer.lower().split())
#          context_words = set(context_text.lower().split())
#          overlap = len(answer_words & context_words)
#          scores["context_relevance"] = min(overlap / len(answer_words), 1.0) if answer_words else 0.0
#  except Exception as e:
#      print(f"Context utilization calculation failed: {e}")
 
 return {}


def improve_query(query: str, context: str = "") -> str:
 """
 Improve query based on context or previous retrievals.
 
 Args:
     query: Original query
     context: Optional context for improvement
     
 Returns:
     Improved query
 """
 
 prompt = f"""This is the original query: {query}. It was unable to retrieve relevant documents from the 
 database. Rewrite the user query so that it can provide more context to the database retrieval.
 The context retrieved from database for the original query: {context}
 """

 res = llm.invoke(prompt)
 return res.content


__all__ = ["reflect_on_answer", "improve_query", "calculate_ragas_metrics"]





 