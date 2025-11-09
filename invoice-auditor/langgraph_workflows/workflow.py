"""
Invoice Processing Workflow using LangGraph with interrupt() and Command.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# Setup logging
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger_config import logger

# Shared checkpointer instance - CRITICAL for resume to work
# All workflow instances must use the same checkpointer to access saved checkpoints
_shared_checkpointer = MemorySaver()

# Local agents
from agents.monitor_agent import check_for_new_invoices
from agents.extractor_agent import extract_invoices_with_llm
from agents.translation_agent import translate_extractions_with_llm
from agents.invoice_data_validation_agent import validate_invoice_data_agent
from agents.business_validation_agent import validate_business_rules_agent
from agents.reporting_agent import generate_reports_with_llm
from agents.rag_agents.indexing_agent import add_chunks_to_db
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class WFState(TypedDict):
    """Simple graph state carried between nodes - processes one file at a time."""
    inbox_items: List[Dict[str, Any]]
    current_item: Dict[str, Any]
    extracted: Dict[str, Any]
    translated_invoice: Dict[str, Any]
    translation_metadata: Dict[str, Any]
    data_validation: Dict[str, Any]
    business_validation: Dict[str, Any]
    all_issues: List[str]
    report: Dict[str, Any]
    processed_count: int
    human_feedback: Dict[str, str]  # basename -> feedback text
    reports: List[Dict[str, Any]]
    needs_human_review: bool


def node_monitor(state: WFState) -> WFState:
    """Monitor: Refresh inbox items."""
    logger.info("📋 Step: Monitor - Checking for new invoices")
    try:
        state["inbox_items"] = check_for_new_invoices()
        count = len(state["inbox_items"])
        logger.info(f"✅ Monitor complete: Found {count} invoice(s) in inbox")
    except Exception as e:
        logger.error(f"❌ Monitor failed: {str(e)}")
        raise
    return state


def node_select_next(state: WFState) -> WFState:
    """Select next file to process."""
    processed = state.get("processed_count", 0)
    total = len(state.get("inbox_items", []))
    
    if processed >= total:
        logger.info("✅ All invoices processed")
        state["current_item"] = {}
        return state

    idx = processed
    current_item = state["inbox_items"][idx]
    basename = current_item.get("basename", "Unknown")
    logger.info(f"📄 Step: Select Next - Processing invoice {idx + 1}/{total}: {basename}")
    
    state["current_item"] = current_item
    state["processed_count"] = idx + 1
    return state


def node_extract(state: WFState) -> WFState:
    """Extract data from current file."""
    current_item = state.get("current_item", {})
    if not current_item:
        state["extracted"] = {}
        return state

    basename = current_item.get("basename", "Unknown")
    logger.info(f"🔍 Step: Extract - Extracting data from {basename}")
    try:
        results = extract_invoices_with_llm([current_item])
        state["extracted"] = results[0] if results else {}
        logger.info(f"✅ Extract complete for {basename}")
    except Exception as e:
        logger.error(f"❌ Extract failed for {basename}: {str(e)}")
        raise
    return state


def node_translate(state: WFState) -> WFState:
    """Translate extracted data to English."""
    extracted = state.get("extracted", {})
    basename = state.get("current_item", {}).get("basename", "Unknown")
    
    if not extracted or not extracted.get("extraction"):
        logger.info(f"⚠️ Translate skipped for {basename}: No extraction data")
        state["translated_invoice"] = {}
        state["translation_metadata"] = {}
        return state

    logger.info(f"🌐 Step: Translate - Translating {basename} to English")
    try:
        results = translate_extractions_with_llm([extracted])
        result = results[0] if results else {}
        
        translation = result.get("translation", {})
        state["translated_invoice"] = translation.get("translated_invoice", {})
        language = translation.get("language", "en")
        state["translation_metadata"] = {
            "language": language,
            "confidence": translation.get("confidence", 1.0),
        }
        logger.info(f"✅ Translate complete for {basename} (from {language})")
    except Exception as e:
        logger.error(f"❌ Translate failed for {basename}: {str(e)}")
        raise
    return state


def node_validate(state: WFState) -> WFState:
    """Run both Invoice Data Validation and Business Validation agents."""
    translated_invoice = state.get("translated_invoice", {})
    basename = state.get("current_item", {}).get("basename", "Unknown")
    
    if not translated_invoice:
        logger.info(f"⚠️ Validate skipped for {basename}: No translated invoice")
        state["data_validation"] = {}
        state["business_validation"] = {}
        state["all_issues"] = []
        return state

    logger.info(f"✔️ Step: Validate - Validating {basename} against ERP records")
    base_url = os.getenv("ERP_BASE_URL", "http://localhost:8000")

    try:
        # Invoice Data Validation
        dv_raw = validate_invoice_data_agent(translated_invoice)
        validation_remarks = {}
        if isinstance(dv_raw, dict):
            validation_remarks = dv_raw.get("validation_remarks", {})

        # Business Validation
        bv_raw = validate_business_rules_agent(translated_invoice, base_url)
        business_validation = {}
        if isinstance(bv_raw, dict):
            business_validation = bv_raw.get("business_validation", {})

        # Combine issues
        dv_notes = []
        if isinstance(validation_remarks, dict):
            dv_notes = validation_remarks.get("basic_notes", []) or []
        bv_summary = []
        if isinstance(business_validation, dict):
            bv_summary = business_validation.get("discrepancy_summary", []) or []
        all_issues = [*dv_notes, *bv_summary]

        state["data_validation"] = validation_remarks
        state["business_validation"] = business_validation
        state["all_issues"] = all_issues
        
        issue_count = len(all_issues)
        if issue_count > 0:
            logger.warning(f"⚠️ Validate complete for {basename}: Found {issue_count} issue(s)")
        else:
            logger.info(f"✅ Validate complete for {basename}: No issues found")
    except Exception as e:
        logger.error(f"❌ Validate failed for {basename}: {str(e)}")
        raise
    return state


def node_report(state: WFState) -> WFState:
    """Generate report for current file."""
    translated_invoice = state.get("translated_invoice", {})
    if not translated_invoice:
        state["report"] = {}
        return state

    basename = state.get("current_item", {}).get("basename", "")
    logger.info(f"📊 Step: Report - Generating report for {basename}")
    
    # Check if human feedback already exists for this file
    existing_feedback = state.get("human_feedback", {}).get(basename, "")
    
    validated_entry = {
        "item": {
            "basename": basename,
            "translation": {
                "translated_invoice": translated_invoice,
                **state.get("translation_metadata", {}),
            }
        },
        "extraction": translated_invoice,
        "issues": state.get("all_issues", []),
        "data_validation": state.get("data_validation", {}),
        "business_validation": state.get("business_validation", {}),
        "human_feedback": existing_feedback,
    }

    try:
        results = generate_reports_with_llm([validated_entry])
        state["report"] = results[0] if results else {}

        # Check if manual review is needed
        recommendation = state["report"].get("recommendation", "")
        state["needs_human_review"] = (recommendation == "Manual Review" and not existing_feedback)

        if state["report"]:
            if "reports" not in state:
                state["reports"] = []
            state["reports"].append(state["report"])
        
        logger.info(f"✅ Report complete for {basename}: Recommendation = {recommendation}")
        if state["needs_human_review"]:
            logger.warning(f"⏸️ {basename} requires manual review")
    except Exception as e:
        logger.error(f"❌ Report generation failed for {basename}: {str(e)}")
        raise
    
    return state


def node_human_feedback(state: WFState) -> WFState:
    """
    Human feedback node - uses interrupt() to pause workflow.
    According to LangGraph docs: When resumed, the node restarts from the beginning,
    and Command(resume=value) becomes the return value of interrupt().
    """
    basename = state.get("current_item", {}).get("basename", "")
    report = state.get("report", {})
    
    # Initialize human_feedback dict if it doesn't exist
    if "human_feedback" not in state:
        state["human_feedback"] = {}
    
    # Prepare data to send to UI
    review_data = {
        "basename": basename,
        "report": report,
        "translated_invoice": state.get("translated_invoice", {}),
        "issues": state.get("all_issues", []),
        "recommendation": report.get("recommendation", "Unknown")
    }
    
    # Pause workflow and wait for user input
    # When resumed with Command(resume=feedback), the feedback becomes the return value
    logger.warning(f"⏸️ Workflow paused: Waiting for human feedback on {basename}")
    feedback = interrupt(review_data)
    
    # After resume, node restarts from beginning and interrupt() returns the resume value
    # Store feedback in state and mark review as complete
    if feedback:
        state["human_feedback"][basename] = feedback
        state["needs_human_review"] = False
        feedback_preview = feedback[:50] + "..." if len(feedback) > 50 else feedback
        logger.info(f"✅ Human feedback received for {basename}: {feedback_preview}")
    else:
        logger.warning(f"⚠️ No feedback provided for {basename}")
    
    return state


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert unsupported types in metadata to strings for Chroma compatibility."""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = str(value)
    return sanitized


def node_index(state: WFState) -> WFState:
    """Create chunks for translated invoice, report, and human feedback, attach metadata, and index."""
    try:
        translated_invoice = state.get("translated_invoice", {})
        current_item = state.get("current_item", {})
        
        if not translated_invoice or not current_item:
            return state
        
        basename = current_item.get("basename", "")
        logger.info(f"📚 Step: Index - Indexing {basename} into vector database")
        
        # Safety check: Don't index if human feedback is required but not provided
        if state.get("needs_human_review", False):
            # Check if feedback exists for this basename
            if basename not in state.get("human_feedback", {}):
                logger.warning(f"⚠️ Skipping index for {basename}: Human feedback required but not provided")
                return state

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)

        # Load metadata.json content for this file
        metadata_dict: Dict[str, Any] = {}
        metadata_path = current_item.get("metadata_path")
        if metadata_path and Path(metadata_path).exists():
            try:
                with open(metadata_path, encoding="utf-8") as mf:
                    metadata_dict = json.load(mf) or {}
            except Exception as e:
                logger.warning(f"⚠️ Could not load metadata for {basename}: {str(e)}")
                metadata_dict = {}

        # Sanitize metadata
        metadata_dict = _sanitize_metadata(metadata_dict)
   
        # Create chunks from translated invoice JSON
        invoice_json_str = json.dumps(translated_invoice, ensure_ascii=False, indent=2)
        invoice_chunks = splitter.split_text(invoice_json_str)
        for chunk_text in invoice_chunks:
            all_chunks.append(Document(
                page_content=chunk_text, 
                metadata={
                    **metadata_dict,
                    "content_type": "invoice_data"
                }
            ))
        
        # Create chunks from report if available
        report = state.get("report", {})
        if report:
            summary = report.get("json", {})
            issues = state.get("all_issues", [])
            report_text_parts = []
            if summary:
                report_text_parts.append("Report Summary:\n" + json.dumps(summary, indent=2))
            if issues:
                report_text_parts.append("Issues:\n" + "\n".join(f"- {it}" for it in issues))
            report_text = "\n\n".join(report_text_parts).strip()
            if report_text:
                rep_doc = Document(
                    page_content=report_text, 
                    metadata={
                        **metadata_dict,
                        "content_type": "report"
                    }
                )
                rep_chunks = splitter.split_documents([rep_doc])
                all_chunks.extend(rep_chunks)

        # Create chunks from human feedback if available
        human_feedback = state.get("human_feedback", {}).get(basename, "")
        if human_feedback:
            feedback_text = f"Human Feedback for {basename}:\n{human_feedback}"
            feedback_doc = Document(
                page_content=feedback_text, 
                metadata={
                    **metadata_dict,
                    "content_type": "human_feedback",
                    "invoice_basename": basename,
                    "has_human_review": "true"
                }
            )
            feedback_chunks = splitter.split_documents([feedback_doc])
            all_chunks.extend(feedback_chunks)
            logger.info(f"✅ Indexed {len(feedback_chunks)} human feedback chunks for {basename}")
        
        # Index all chunks
        if all_chunks:
            add_chunks_to_db(all_chunks)
            logger.info(f"✅ Index complete for {basename}: {len(all_chunks)} chunk(s) indexed")
        else:
            logger.warning(f"⚠️ No chunks to index for {basename}")

    except Exception as e:
        logger.error(f"❌ Indexing error for {basename}: {str(e)}")
    
    return state


def _check_recommendation(state: WFState) -> str:
    """Conditional edge: check if human review is needed."""
    if state.get("needs_human_review", False):
        return "human_feedback"
    return "index"


def _should_continue(state: WFState) -> str:
    """Conditional edge: continue processing if more files exist."""
    processed = state.get("processed_count", 0)
    total = len(state.get("inbox_items", []))
    if processed < total:
        return "process_next"
    return "end"


def build_workflow():
    """Build workflow with interrupt support for human feedback."""
    graph = StateGraph(WFState)

    # Add nodes
    graph.add_node("monitor", node_monitor)
    graph.add_node("select_next", node_select_next)
    graph.add_node("extract", node_extract)
    graph.add_node("translate", node_translate)
    graph.add_node("validate", node_validate)
    graph.add_node("report", node_report)
    graph.add_node("human_feedback", node_human_feedback)
    graph.add_node("index", node_index)

    # Flow
    graph.set_entry_point("monitor")
    graph.add_edge("monitor", "select_next")
    graph.add_edge("select_next", "extract")
    graph.add_edge("extract", "translate")
    graph.add_edge("translate", "validate")
    graph.add_edge("validate", "report")
    graph.add_conditional_edges("report", _check_recommendation, {
        "human_feedback": "human_feedback",
        "index": "index",
    })
    graph.add_edge("human_feedback", "index")
    graph.add_conditional_edges("index", _should_continue, {
        "process_next": "select_next",
        "end": END,
    })

    # Compile with shared checkpointer for interrupt support
    # Using shared checkpointer ensures checkpoints persist across workflow instances
    return graph.compile(checkpointer=_shared_checkpointer)


def run_workflow_with_checkpoints(thread_id: str, human_feedback: Optional[Dict[str, str]] = None):
    """
    Run workflow with checkpoint support for interrupts.
    
    Args:
        thread_id: Unique identifier for this workflow run
        human_feedback: Optional feedback to resume interrupted workflow
    
    Yields:
        Events from the workflow execution
    """
    logger.info(f"🚀 Starting workflow execution - Thread ID: {thread_id}")
    if human_feedback:
        logger.info(f"📝 Resuming with {len(human_feedback)} feedback entries")
    
    wf = build_workflow()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state: WFState = {
        "inbox_items": [],
        "current_item": {},
        "extracted": {},
        "translated_invoice": {},
        "translation_metadata": {},
        "data_validation": {},
        "business_validation": {},
        "all_issues": [],
        "report": {},
        "processed_count": 0,
        "human_feedback": human_feedback or {},
        "reports": [],
        "needs_human_review": False,
    }
    
    try:
        # Stream events
        for event in wf.stream(initial_state, config, stream_mode="updates"):
            yield event
    except Exception as e:
        logger.error(f"❌ Workflow execution failed - Thread ID: {thread_id}, Error: {str(e)}")
        raise


def resume_workflow(thread_id: str, feedback: str, basename: str):
    """
    Resume an interrupted workflow with human feedback.
    
    Args:
        thread_id: The thread_id of the interrupted workflow
        feedback: Human feedback text
        basename: Invoice basename being reviewed
    
    Returns:
        Generator of events from resuming the workflow
    """
    from langgraph.types import Command
    
    feedback_preview = feedback[:50] + "..." if len(feedback) > 50 else feedback
    logger.info(f"▶️ Resuming workflow - Thread ID: {thread_id}, Invoice: {basename}, Feedback: {feedback_preview}")
    
    wf = build_workflow()
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get current checkpointed state
        current_state = wf.get_state(config)
        if not current_state.values:
            logger.error(f"❌ No checkpointed state found for thread {thread_id}")
            raise RuntimeError(f"No checkpointed state found for thread {thread_id}")
        
        logger.info(f"📋 Current checkpoint state: processed_count={current_state.values.get('processed_count', 0)}, needs_human_review={current_state.values.get('needs_human_review', False)}")
        
        # According to LangGraph docs: Command(resume=value) passes value to interrupt() return
        # The node will restart from beginning, and interrupt() will return this feedback value
        logger.info(f"📝 Resuming with feedback for {basename}...")
        
        for event in wf.stream(
            Command(resume=feedback),
            config,
            stream_mode="updates"
        ):
            for step_name, step_output in event.items():
                if not step_name.startswith("__"):
                    logger.info(f"🔄 Resumed workflow executing: {step_name}")
            yield event
        
        logger.info(f"✅ Workflow resumed successfully - Thread ID: {thread_id}, Invoice: {basename}")
    except Exception as e:
        logger.error(f"❌ Failed to resume workflow - Thread ID: {thread_id}, Invoice: {basename}, Error: {str(e)}", exc_info=True)
        raise


__all__ = [
    "build_workflow", 
    "run_workflow_with_checkpoints", 
    "resume_workflow",
    "WFState"
]