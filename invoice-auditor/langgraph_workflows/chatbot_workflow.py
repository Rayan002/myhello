import warnings
warnings.filterwarnings("ignore")

import os
import sys
import sqlite3
import uuid
import json
from pathlib import Path
from typing import Annotated, TypedDict, List, Tuple, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger_config import logger

from agents.rag_agents.retrieval_agent import retrieve_with_metadata
from agents.rag_agents.augmentation_agent import format_context, grade_documents
from agents.rag_agents.reflection_agent import reflect_on_answer, improve_query, calculate_ragas_metrics
from agents.rag_agents.generation_agent import generate_answer

load_dotenv()

_thread_title_file = Path(__file__).resolve().parent.parent / "thread_title.json"

def _load_thread_titles() -> Dict[str, Any]:
    if _thread_title_file.exists():
        try:
            with open(_thread_title_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and data and isinstance(list(data.values())[0], dict):
                    return data
                return {tid: {"title": title, "timestamp": 0} if isinstance(title, str) else title for tid, title in data.items()}
        except Exception:
            return {}
    return {}

def _save_thread_title(thread_id: str, title: str):
    import time
    titles = _load_thread_titles()
    titles[thread_id] = {"title": title, "timestamp": time.time()}
    try:
        with open(_thread_title_file, 'w') as f:
            json.dump(titles, f, indent=2)
    except Exception:
        pass

def _get_thread_title(thread_id: str) -> str:
    titles = _load_thread_titles()
    title_data = titles.get(thread_id, {})
    if isinstance(title_data, dict):
        return title_data.get("title", thread_id[:8])
    return title_data if isinstance(title_data, str) else thread_id[:8]


class ChatState(TypedDict):
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
        logger.warning("🔍 Retrieve: No human message found, skipping retrieval")
        return {"docs": [], "context_text": ""}

    question = last_message.content
    logger.info(f"🔍 Retrieve: Searching for context for question: {question[:100]}...")
    docs = retrieve_with_metadata(question, k=6)
    doc_count = len(docs) if docs else 0
    logger.info(f"✅ Retrieve: Found {doc_count} relevant document(s)")
    context_text = format_context(docs) if docs else ""
    return {"docs": docs, "context_text": context_text}


 

def node_augment(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, HumanMessage):
        logger.warning("📊 Augment: No human message found, marking as bad")
        return {"augment_decision": {"decision": "bad"}}

    question = last_message.content
    docs = state.get("docs", [])
    logger.info(f"📊 Augment: Grading {len(docs)} document(s) for relevance")
    decision = grade_documents(question, docs)
    decision_result = decision.get("decision", "unknown")
    logger.info(f"📊 Augment: Decision = {decision_result}")
    return {"augment_decision": decision}


 

def node_transform_retrieve(state: ChatState):
    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        logger.warning("🔄 Transform Retrieve: No human message found")
        return {}

    original_question = last_message.content
    logger.info(f"🔄 Transform Retrieve: Improving query: {original_question[:100]}...")
    improved = improve_query(original_question, context=state['context_text'])
    logger.info(f"🔄 Transform Retrieve: Improved query: {improved[:100]}...")
    docs = retrieve_with_metadata(improved, k=6)
    doc_count = len(docs) if docs else 0
    logger.info(f"✅ Transform Retrieve: Found {doc_count} document(s) with improved query")
    context_text = format_context(docs) if docs else ""
    return {"docs": docs, "context_text": context_text}



def node_generate(state: ChatState):
    q = state["messages"][-1].content
    ctx = state.get("context_text", "")
    docs = state.get("docs", [])
    is_regenerate = state.get("regenerate", False)

    hint = ""
    if is_regenerate:
        hint = "\nNote: The previous answer was not relevant. Focus strictly on the user's question and cite matching context."
        logger.info("🔄 Generate: Regenerating answer with improved focus")
    
    q_effective = f"{q}{hint}" if hint else q
    logger.info(f"💬 Generate: Generating answer for question: {q[:100]}...")
    logger.info(f"💬 Generate: Using context length: {len(ctx)} characters, {len(docs)} documents")

    try:
        answer = generate_answer(q_effective, ctx)
        logger.info(f"✅ Generate: Answer generated (length: {len(answer)} characters)")
    except Exception as e:
        logger.error(f"❌ Generate: Error generating answer: {str(e)}")
        answer = f"Error generating answer: {str(e)}"

    general_response = "I'm an AI Invoice Auditor Assistant specialized in invoice analysis"
    if general_response in answer and not ctx:
        metrics = {"groundedness": 0.0, "relevance": 0.5, "context_relevance": 0.0}
        logger.warning("⚠️ Generate: General response detected, using default metrics")
    else:
        contexts = [ctx] if ctx else []
        if docs:
            contexts = [d.get("page_content", "") for d in docs[:5]]
        metrics = calculate_ragas_metrics(q, answer, contexts, docs)
        logger.info(f"📊 Generate: RAGAS metrics - Relevance: {metrics.get('relevance', 0):.2%}, Groundedness: {metrics.get('groundedness', 0):.2%}, Context Relevance: {metrics.get('context_relevance', 0):.2%}")
    
    res = AIMessage(content=answer, additional_kwargs={"rag_metrics": metrics})
    return {"answer_text": answer, "messages": [res], "question": q, "metrics": metrics}


 

def node_reflect(state: ChatState):
    attempts = state.get("attempts", 0)
    answer = state.get("answer_text", "")
    context_text = state.get("context_text", "")
    question = state.get("question", "")

    logger.info(f"🤔 Reflect: Evaluating answer quality (attempt {attempts + 1})")
    reflection = reflect_on_answer(answer, context_text, question)
    logger.info(f"🤔 Reflect: Reflection result = {reflection}")

    if reflection.lower() == "bad" and attempts < 1:
        logger.warning(f"⚠️ Reflect: Answer quality is bad, will regenerate (attempt {attempts + 1})")
        return {"attempts": attempts + 1, "regenerate": True}

    logger.info("✅ Reflect: Answer quality is acceptable, proceeding to end")
    return {"regenerate": False}


def _reflect_decider(s: ChatState):
    regenerate = s.get("regenerate", False)
    next_step = "generate" if regenerate else "end"
    if regenerate:
        logger.info("🔄 Reflect Decider: Regenerating answer")
    else:
        logger.info("✅ Reflect Decider: Answer accepted, ending workflow")
    return next_step

def _aug_decider(s: ChatState):
    d = s.get("augment_decision", {})
    decision = d.get("decision", "unknown")
    next_step = "generate" if decision == "good" else "transform_retrieve"
    if decision == "good":
        logger.info("✅ Augment Decider: Documents are good, proceeding to generate")
    else:
        logger.info("🔄 Augment Decider: Documents need improvement, transforming query")
    return next_step


 

_db_path = os.path.join(os.path.dirname(__file__).replace("langgraph_workflows", ""), "chatbot.db")
conn = sqlite3.connect(_db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)


 

def build_chatbot():
    logger.info("🔧 Building chatbot workflow graph")
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
    logger.info("✅ Chatbot workflow graph compiled successfully")
    return compiled


 

def retrieve_all_threads():
    logger.info("📋 Retrieving all chat threads")
    all_threads = []
    titles = _load_thread_titles()
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                title_data = titles.get(thread_id, {})
                if isinstance(title_data, dict):
                    timestamp = title_data.get("timestamp", 0)
                else:
                    timestamp = 0
                all_threads.append((thread_id, timestamp))
    except Exception as e:
        logger.error(f"❌ Error retrieving threads: {str(e)}")
    
    if not all_threads:
        logger.info("📋 No threads found")
        return []
    
    all_threads.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"📋 Found {len(all_threads)} thread(s)")
    return [tid for tid, _ in all_threads]


 

def generate_thread_id():
    thread_id = str(uuid.uuid4())
    logger.info(f"🆔 Generated new thread ID: {thread_id}")
    return thread_id


 

def get_thread_preview(thread_id: str) -> str:
    stored_title = _get_thread_title(thread_id)
    if stored_title != thread_id[:8]:
        if len(stored_title) > 30:
            return stored_title[:27] + "..."
        return stored_title
    
    try:
        state = _chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        
        for m in messages:
            if isinstance(m, HumanMessage):
                txt = (m.content or "").strip()
                if txt:
                    title = txt[:30] if len(txt) <= 30 else txt[:27] + "..."
                    _save_thread_title(thread_id, title)
                    logger.info(f"📝 Generated thread title for {thread_id[:8]}: {title}")
                    return title
    except Exception as e:
        logger.warning(f"⚠️ Error getting thread preview for {thread_id[:8]}: {str(e)}")
    
    return thread_id[:8]


 

def retrieve_threads_with_titles() -> List[Tuple[str, str]]:
    ids = retrieve_all_threads()
    return [(tid, get_thread_preview(tid)) for tid in ids]


 

def save_thread_title(thread_id: str, title: str):
    logger.info(f"💾 Saving thread title for {thread_id[:8]}: {title}")
    _save_thread_title(thread_id, title)

def update_thread_timestamp(thread_id: str):
    import time
    logger.debug(f"🕐 Updating timestamp for thread {thread_id[:8]}")
    titles = _load_thread_titles()
    if thread_id in titles:
        title_data = titles[thread_id]
        if isinstance(title_data, dict):
            titles[thread_id] = {"title": title_data.get("title", ""), "timestamp": time.time()}
        else:
            titles[thread_id] = {"title": title_data if isinstance(title_data, str) else "", "timestamp": time.time()}
        try:
            with open(_thread_title_file, 'w') as f:
                json.dump(titles, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error updating thread timestamp: {str(e)}")

__all__ = [
   "build_chatbot",
   "retrieve_all_threads",
   "retrieve_threads_with_titles",
   "generate_thread_id",
   "get_thread_preview",
   "save_thread_title",
   "update_thread_timestamp",
   "ChatState",
]



