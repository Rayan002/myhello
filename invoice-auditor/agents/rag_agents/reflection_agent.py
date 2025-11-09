"""
Reflection Agent: Reflects on RAG retrieval results and improves query/answer quality.
"""
import os
from typing import Dict, List, Any

from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from dotenv import load_dotenv

load_dotenv()


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

   from langchain_google_genai import ChatGoogleGenerativeAI
   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
   
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


def calculate_ragas_metrics(question: str, answer: str, contexts: List[str], docs: List[Dict[str, Any]]) -> Dict[str, float]:
   """
   Calculate RAGAS metrics using built-in evaluators.
   
   Better approach: Uses RAGAS's optimized ContextRelevancy metric 
   instead of manual LLM prompting.
   """
#    from ragas import evaluate, EvaluationDataset
#    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevancy
#    from ragas.llms import LangchainLLMWrapper
#    from langchain_google_genai import ChatGoogleGenerativeAI
   
#    # Initialize LLM once
#    evaluator_llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#    evaluator_llm = LangchainLLMWrapper(evaluator_llm_instance)
   
#    # Prepare data for RAGAS
#    dataset_list = [{
#        "user_input": question,
#        "retrieved_contexts": contexts if contexts else [""],
#        "response": answer,
#    }]
   
#    evaluation_dataset = EvaluationDataset.from_list(dataset_list)
   
#    # Use built-in metrics - no manual LLM prompting needed
#    result = evaluate(
#        dataset=evaluation_dataset,
#        metrics=[
#            Faithfulness(),
#            AnswerRelevancy(),
#            ContextRelevancy()  # Built-in metric (better than manual)
#        ],
#        llm=evaluator_llm,
#        raise_exceptions=False,
#    )
   
   scores = {}
   
   return {
       "groundedness": float(scores.get("faithfulness", [0.0])[0]) if scores.get("faithfulness") else 0.0,
       "relevance": float(scores.get("answer_relevancy", [0.0])[0]) if scores.get("answer_relevancy") else 0.0,
       "context_relevance": float(scores.get("context_relevancy", [0.0])[0]) if scores.get("context_relevancy") else 0.0,
   }


def improve_query(query: str, context: str = "") -> str:
   """
   Improve query based on context or previous retrievals.
   
   Args:
       query: Original query
       context: Optional context for improvement
       
   Returns:
       Improved query
   """
   from langchain_google_genai import ChatGoogleGenerativeAI
   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
   
   prompt = f"""This is the original query: {query}. It was unable to retrieve relevant documents from the 
   database. Rewrite the user query so that it can provide more context to the database retrieval.
   The context retrieved from database for the original query: {context}
   """

   res = llm.invoke(prompt)
   return res.content


__all__ = ["reflect_on_answer", "improve_query", "calculate_ragas_metrics"]