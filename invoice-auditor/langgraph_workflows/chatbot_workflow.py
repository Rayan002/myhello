"""
RAG Chatbot Workflow using LangGraph with persistent memory.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
import uuid
import json
from pathlib import Path
from typing import Annotated, TypedDict, List, Tuple, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # type: ignore
from langgraph.checkpoint.sqlite import SqliteSaver # type: ignore


from agents.rag_agents.retrieval_agent import retrieve_with_metadata
from agents.rag_agents.augmentation_agent import format_context, grade_documents
from agents.rag_agents.reflection_agent import reflect_on_answer, improve_query, calculate_ragas_metrics
from agents.rag_agents.generation_agent import generate_answer

load_dotenv()

# Thread title storage
_thread_title_file = Path(__file__).resolve().parent.parent / "thread_title.json"

def _load_thread_titles() -> Dict[str, str]:
   """Load thread titles from JSON file."""
   if _thread_title_file.exists():
       try:
           with open(_thread_title_file, 'r') as f:
               return json.load(f)
       except Exception:
           return {}
   return {}

def _save_thread_title(thread_id: str, title: str):
   """Save thread title to JSON file."""
   titles = _load_thread_titles()
   titles[thread_id] = title
   try:
       with open(_thread_title_file, 'w') as f:
           json.dump(titles, f, indent=2)
   except Exception:
       pass

def _get_thread_title(thread_id: str) -> str:
   """Get thread title from storage."""
   titles = _load_thread_titles()
   return titles.get(thread_id, thread_id[:8])


class ChatState(TypedDict):

   """Chatbot state with messages and intermediate RAG data."""

   messages: Annotated[list[BaseMessage], add_messages]

   docs: List[Dict[str, Any]]

   context_text: str

   answer_text: str

   attempts: int

   augment_decision: Dict[str, Any]

   regenerate: bool

   metrics: Dict[str, float]

   question: str


 

def node_retrieve(state: ChatState):

   messages = state["messages"]

   last_message = messages[-1] if messages else None

   if not isinstance(last_message, HumanMessage):

       return {"docs": [], "context_text": ""}

   question = last_message.content

   docs = retrieve_with_metadata(question, k=6)

   return {"docs": docs, "context_text": format_context(docs) if docs else ""}


 

def node_augment(state: ChatState):

   messages = state["messages"]

   last_message = messages[-1] if messages else None

   if not isinstance(last_message, HumanMessage):

       return {"augment_decision": {"decision": "bad"}}

   question = last_message.content

   decision = grade_documents(question, state.get("docs", []))

   return {"augment_decision": decision}


 

def node_transform_retrieve(state: ChatState):

   last_message = state["messages"][-1]

   if not isinstance(last_message, HumanMessage):
       return {}

   improved = improve_query(last_message.content, context=state['context_text'])

   docs = retrieve_with_metadata(improved, k=6)

   return {"docs": docs, "context_text": format_context(docs) if docs else ""}



def node_generate(state: ChatState):
   q = state["messages"][-1].content
   ctx = state.get("context_text", "")
   docs = state.get("docs", [])

   # Include a hint when regenerating
   hint = ""
   if state.get("regenerate", False):
       hint = "\nNote: The previous answer was not relevant. Focus strictly on the user's question and cite matching context."
   
   q_effective = f"{q}{hint}" if hint else q

   # Use generation agent with context we already retrieved (handles general questions internally)
   try:
       answer = generate_answer(q_effective, ctx)
   except Exception as e:
       answer = f"Error generating answer: {str(e)}"

   # Calculate RAGAS metrics in generate node
   # Check if answer is the general response (handled by generation_agent via prompt guardrails)
   general_response = "I'm an AI Invoice Auditor Assistant specialized in invoice analysis"
   if general_response in answer and not ctx:
       metrics = {"groundedness": 0.0, "relevance": 0.5, "context_relevance": 0.0}
   else:
       contexts = [ctx] if ctx else []
       if docs:
           contexts = [d.get("page_content", "") for d in docs[:5]]
       metrics = calculate_ragas_metrics(q, answer, contexts, docs)
   
   # Store metrics in AIMessage additional_kwargs
   res = AIMessage(content=answer, additional_kwargs={"rag_metrics": metrics})

   return {"answer_text": answer, "messages": [res], "question": q, "metrics": metrics}


 

def node_reflect(state: ChatState):
   attempts = state.get("attempts", 0)
   answer = state.get("answer_text", "")
   context_text = state.get("context_text", "")
   question = state.get("question", "")

   # Reflection for regeneration only (no metrics calculation here)
   reflection = reflect_on_answer(answer, context_text, question)

   if reflection.lower() == "bad" and attempts < 1:
       return {"attempts": attempts + 1, "regenerate": True}

   return {"regenerate": False}


## conditionals

def _reflect_decider(s: ChatState):

   return "generate" if s.get("regenerate") else "end"

def _aug_decider(s: ChatState):

   d = s.get("augment_decision", {})

   return "generate" if d.get("decision") == "good" else "transform_retrieve"


 

# Initialize checkpointer (persistent memory)

_db_path = os.path.join(os.path.dirname(__file__).replace("langgraph_workflows", ""), "chatbot.db")

conn = sqlite3.connect(_db_path, check_same_thread=False)

checkpointer = SqliteSaver(conn)


 

def build_chatbot():

   """Build the RAG chatbot workflow."""

   graph = StateGraph(ChatState)

   graph.add_node("retrieve", node_retrieve)

   graph.add_node("augment", node_augment)

   graph.add_node("transform_retrieve", node_transform_retrieve)

   graph.add_node("generate", node_generate)

   graph.add_node("reflect", node_reflect)

   graph.set_entry_point("retrieve")

   graph.add_edge("retrieve", "augment")

   graph.add_conditional_edges("augment", _aug_decider, {"generate": "generate", "transform_retrieve": "transform_retrieve"})

   graph.add_edge("transform_retrieve", "generate")

   graph.add_edge("generate", "reflect")

   graph.add_conditional_edges("reflect", _reflect_decider, {"generate": "generate", "end": END})

   compiled = graph.compile(checkpointer=checkpointer)

   global _chatbot

   _chatbot = compiled

   return compiled


 

def retrieve_all_threads():

   """Retrieve all conversation thread IDs."""

   all_threads = set()

   try:

       for checkpoint in checkpointer.list(None):

           thread_id = checkpoint.config.get("configurable", {}).get("thread_id")

           if thread_id:

               all_threads.add(thread_id)

   except Exception:

       pass

   return list(all_threads)


 

def generate_thread_id():

   """Generate a new thread ID."""

   return str(uuid.uuid4())


 

def get_thread_preview(thread_id: str) -> str:
   """Return thread title from storage or generate from first message."""
   # First try to get from storage
   stored_title = _get_thread_title(thread_id)
   if stored_title != thread_id[:8]:
       return stored_title
   
   # If not in storage, get from first message and save it
   try:
       state = _chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
       messages = state.values.get("messages", [])
       
       for m in messages:
           if isinstance(m, HumanMessage):
               txt = (m.content or "").strip()
               if txt:
                   # Save the full question as title (or first 50 chars)
                   title = txt[:50] if len(txt) <= 50 else txt[:47] + "..."
                   _save_thread_title(thread_id, title)
                   return title
   except Exception:
       pass
   
   return thread_id[:8]


 

def retrieve_threads_with_titles() -> List[Tuple[str, str]]:
   ids = retrieve_all_threads()
   return [(tid, get_thread_preview(tid)) for tid in ids]


 

def save_thread_title(thread_id: str, title: str):
   """Public function to save thread title."""
   _save_thread_title(thread_id, title)

__all__ = [
   "build_chatbot",
   "retrieve_all_threads",
   "retrieve_threads_with_titles",
   "generate_thread_id",
   "get_thread_preview",
   "save_thread_title",
   "ChatState",
]



