"""
Process Invoices Page - Using interrupt() and Command for human feedback
"""
import os
import sys
from pathlib import Path
import streamlit as st
import time
import uuid

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph_workflows.workflow import run_workflow_with_checkpoints, resume_workflow
from agents.monitor_agent import get_incoming_dir, peek_new_basenames, save_uploaded_files
from utils.logger_config import logger

# Page configuration
st.set_page_config(
    page_title="Process Invoices - AI Invoice Auditor",
    layout="wide"
)

st.title("Process Invoices")
st.markdown("Run the invoice processing workflow to extract, translate, validate, and generate reports.")
st.markdown("---")

# Upload to incoming directory
st.subheader("Add Documents to Inbox")
uploads = st.file_uploader(
    "Upload invoice files (pdf, docx, images) and corresponding metadata JSON",
    type=["pdf", "docx", "png", "jpg", "jpeg", "json"],
    accept_multiple_files=True,
)

col_up1, col_up2 = st.columns([1, 1])
with col_up1:
    save_btn = st.button("Save to Inbox", use_container_width=True)
    if save_btn:
        if uploads:
            errs: list[str] = []
            saved = save_uploaded_files(uploads, errs)
            if saved:
                logger.info(f"📁 Files saved to inbox: {', '.join(sorted(set(saved)))}")
                st.success(f"Saved: {', '.join(sorted(set(saved)))}")
            if errs:
                for e in errs:
                    logger.error(f"❌ Error saving file: {e}")
                    st.warning(e)
        else:
            st.warning("Please select files to upload first.")
with col_up2:
    if st.button("Show New Files", use_container_width=True):
        try:
            new_list = peek_new_basenames()
            if new_list:
                st.info("New files detected: " + ", ".join(new_list))
            else:
                st.info("No new files in inbox.")
        except Exception as ex:
            st.error(f"Error reading inbox: {ex}")

inbox_path = get_incoming_dir()
st.caption(f"Inbox directory: {inbox_path}")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "reports" not in st.session_state:
    st.session_state.reports = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "interrupted_items" not in st.session_state:
    st.session_state.interrupted_items = []  # Items waiting for feedback
if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = {}

# Process button
process_btn = st.button("Start Processing", type="primary", disabled=st.session_state.is_processing)

# Check if workflow is already interrupted - don't re-run if waiting for feedback
has_interrupted_items = len(st.session_state.interrupted_items) > 0

if process_btn:
    # Only start processing if button is clicked and not already interrupted
    logger.info("🚀 User initiated invoice processing workflow")
    st.session_state.is_processing = True
    st.session_state.interrupted_items = []  # Reset interrupted items when starting fresh
    
if st.session_state.is_processing and not has_interrupted_items:
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_details = st.empty()
    
    # Status tracking
    steps = ["monitor", "extract", "translate", "validate", "report", "human_feedback", "index"]
    step_names = {
        "monitor": "Monitoring for new invoices...",
        "extract": "Extracting invoice data...",
        "translate": "Translating to English...",
        "validate": "Validating against ERP records...",
        "report": "Generating reports...",
        "human_feedback": "Requesting human feedback...",
        "index": "Indexing documents...",
        "select_next": "Selecting next invoice...",
    }
    
    processed_files = []
    reports_collected = []
    workflow_interrupted = False
    current_file_basename = None
    current_step = None
    
    try:
        # Run workflow with checkpoints
        for event in run_workflow_with_checkpoints(
            thread_id=st.session_state.thread_id,
            human_feedback=st.session_state.human_feedback
        ):
            # Check if this is an interrupt event FIRST - before processing any other events
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]
                if len(interrupt_data) > 0:
                    # Workflow paused for human feedback
                    review_data = interrupt_data[0].value
                    
                    basename = review_data.get("basename", "Unknown")
                    report = review_data.get("report", {})
                    
                    # Store interrupted item
                    st.session_state.interrupted_items.append({
                        "basename": basename,
                        "review_data": review_data,
                        "report": report
                    })
                    
                    # Add to reports for display
                    if report not in reports_collected:
                        reports_collected.append(report)
                    
                    workflow_interrupted = True
                    logger.warning(f"⏸️ Workflow interrupted: {basename} requires manual review")
                    status_text.warning(f"⏸️ Workflow paused: {basename} requires manual review")
                    step_details.info("Please provide feedback below to continue processing.")
                    # IMPORTANT: Break immediately - don't process any more events
                    break
            
            # Process normal events ONLY if not interrupted
            # Skip ALL processing if we detected an interrupt in this iteration
            if not workflow_interrupted:
                for step, output in event.items():
                    if step.startswith("__"):
                        continue
                    
                    # Skip index step entirely - it should only run after resume
                    if step == "index":
                        continue
                    
                    # Track current step
                    current_step = step
                    
                    # Get current file being processed from state
                    if isinstance(output, dict):
                        # Always check for current_item in the output (which is the state after the node)
                        current_item = output.get("current_item", {})
                        if current_item:
                            new_basename = current_item.get("basename", "Unknown")
                            if new_basename and new_basename != "Unknown":
                                current_file_basename = new_basename
                                if current_file_basename not in processed_files:
                                    processed_files.append(current_file_basename)
                    
                    # Update progress
                    if step in steps:
                        step_idx = steps.index(step)
                        progress = (step_idx + 1) / len(steps)
                        progress_bar.progress(progress)
                        
                        # Show current step with file name
                        step_display = step_names.get(step, step)
                        if current_file_basename and current_file_basename != "Unknown":
                            status_text.info(f"Current Step: {step_display} - {current_file_basename}")
                            step_details.success(f"Processing: {current_file_basename}")
                        else:
                            status_text.info(f"Current Step: {step_display}")
                    
                    # Collect reports
                    if step == "report":
                        if isinstance(output, dict):
                            report = output.get("report", {})
                            if report and report not in reports_collected:
                                reports_collected.append(report)
                                
                                summary = report.get("json", {})
                                recommendation = report.get("recommendation", "Unknown")
                                basename = summary.get("basename", "Unknown")
                                
                                step_details.success(f"{basename} - Recommendation: {recommendation}")
        
        # Update session state reports
        if reports_collected:
            existing_basenames = {r.get("json", {}).get("basename") for r in st.session_state.reports if isinstance(r, dict)}
            for r in reports_collected:
                r_basename = r.get("json", {}).get("basename")
                if r_basename not in existing_basenames:
                    st.session_state.reports.append(r)
                else:
                    # Update existing report
                    for idx, existing_r in enumerate(st.session_state.reports):
                        if isinstance(existing_r, dict) and existing_r.get("json", {}).get("basename") == r_basename:
                            st.session_state.reports[idx] = r
                            break
        
        if not workflow_interrupted:
            logger.info("✅ Invoice processing workflow completed successfully")
            progress_bar.progress(1.0)
            status_text.success("✅ Processing completed successfully!")
            step_details.empty()
            st.session_state.is_processing = False
        
    except Exception as e:
        logger.error(f"❌ Error during invoice processing: {str(e)}", exc_info=True)
        status_text.error(f"❌ Error during processing: {str(e)}")
        step_details.error("Please check the Debug Mode for detailed error information.")
        st.session_state.is_processing = False
elif has_interrupted_items:
    # Show status when waiting for feedback
    status_text = st.empty()
    status_text.warning(f"⏸️ Workflow paused: Waiting for human feedback on {len(st.session_state.interrupted_items)} invoice(s)")

st.markdown("---")

# Display Results and Handle Feedback
if st.session_state.reports:
    st.header("Processing Results")
    
    for idx, report in enumerate(st.session_state.reports, 1):
        if not isinstance(report, dict):
            continue
        
        summary = report.get("json", {})
        basename = summary.get("basename", f"Invoice {idx}")
        recommendation = report.get("recommendation", "Unknown")
        
        # Check if this item is waiting for feedback
        is_interrupted = any(item["basename"] == basename for item in st.session_state.interrupted_items)
        
        # Report card
        with st.container(border=True):
            # Header
            col_title, col_status = st.columns([3, 1])
            with col_title:
                st.subheader(basename)
                if is_interrupted:
                    st.caption("⏸️ Workflow paused - waiting for your feedback")
            with col_status:
                if recommendation == "Approve":
                    st.success("✅ Approved")
                elif recommendation == "Reject":
                    st.error("❌ Rejected")
                else:
                    st.warning("⏳ Manual Review")
            
            # Two-column layout
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.subheader("Full Report")
                html_report = report.get("report", "")
                if html_report:
                    try:
                        import streamlit.components.v1 as components
                        components.html(html_report, height=700, scrolling=True)
                    except Exception:
                        st.markdown(html_report, unsafe_allow_html=True)
                else:
                    st.info("No HTML report available")
            
            with col_right:
                # Issues and discrepancies
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("Issues Found")
                    issues = summary.get("issues", [])
                    if issues:
                        for issue in issues:
                            st.error(f"• {issue}")
                    else:
                        st.success("No issues found")
                
                with col_b:
                    st.subheader("Discrepancies")
                    discrepancies = summary.get("discrepancy_summary", [])
                    if discrepancies:
                        for disc in discrepancies:
                            st.warning(f"• {disc}")
                    else:
                        st.success("No discrepancies")
                
                # Missing fields
                missing_fields = summary.get("missing_fields", [])
                if missing_fields:
                    st.subheader("Missing Fields")
                    st.error(", ".join(missing_fields))
            
            # Human feedback section
            with col_right:
                st.subheader("Human Feedback")
                
                # Check if feedback already provided
                existing_feedback = st.session_state.human_feedback.get(basename, "")
                
                if existing_feedback:
                    st.success("✅ Feedback Provided:")
                    st.info(existing_feedback)
                
                elif is_interrupted:
                    st.warning("⚠️ This invoice is waiting for your review to continue processing.")
                    
                    # Feedback input
                    feedback = st.text_area(
                        "Enter your review notes or decision rationale",
                        key=f"feedback_{basename}",
                        height=150,
                    )
                    
                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("✅ Approve & Resume", key=f"approve_{basename}", use_container_width=True, type="primary"):
                            approval_text = f"[HUMAN APPROVED] {feedback}" if feedback.strip() else "[HUMAN APPROVED] No additional comments."
                            
                            logger.info(f"✅ User approved invoice: {basename}")
                            if feedback.strip():
                                logger.info(f"📝 Approval feedback: {feedback[:100]}...")
                            
                            # Store feedback
                            st.session_state.human_feedback[basename] = approval_text
                            
                            # Remove from interrupted items BEFORE resuming
                            st.session_state.interrupted_items = [
                                item for item in st.session_state.interrupted_items 
                                if item["basename"] != basename
                            ]
                            
                            # Resume workflow with Command
                            with st.spinner("Resuming workflow..."):
                                try:
                                    reports_from_resume = []
                                    # Process streamed events from resumed workflow
                                    for event in resume_workflow(
                                        thread_id=st.session_state.thread_id,
                                        feedback=approval_text,
                                        basename=basename
                                    ):
                                        # Check for new interrupts
                                        if "__interrupt__" in event:
                                            # Another interrupt occurred
                                            interrupt_data = event["__interrupt__"]
                                            if len(interrupt_data) > 0:
                                                review_data = interrupt_data[0].value
                                                new_basename = review_data.get("basename", "Unknown")
                                                st.session_state.interrupted_items.append({
                                                    "basename": new_basename,
                                                    "review_data": review_data,
                                                    "report": review_data.get("report", {})
                                                })
                                        
                                        # Collect reports from resumed workflow
                                        for step, output in event.items():
                                            if step.startswith("__"):
                                                continue
                                            if step == "report" and isinstance(output, dict):
                                                report = output.get("report", {})
                                                if report:
                                                    reports_from_resume.append(report)
                                    
                                    # Update reports in session state
                                    for report in reports_from_resume:
                                        r_basename = report.get("json", {}).get("basename")
                                        if r_basename:
                                            found = False
                                            for idx, existing_r in enumerate(st.session_state.reports):
                                                if isinstance(existing_r, dict) and existing_r.get("json", {}).get("basename") == r_basename:
                                                    st.session_state.reports[idx] = report
                                                    found = True
                                                    break
                                            if not found:
                                                st.session_state.reports.append(report)
                                    
                                    # Check if workflow is completely done
                                    if len(st.session_state.interrupted_items) == 0:
                                        st.session_state.is_processing = False
                                    
                                    st.success(f"✅ Workflow resumed! {basename} approved and indexed.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error resuming workflow: {e}")
                    
                    with col_btn2:
                        if st.button("❌ Reject & Resume", key=f"reject_{basename}", use_container_width=True):
                            rejection_text = f"[HUMAN REJECTED] {feedback}" if feedback.strip() else "[HUMAN REJECTED] No additional comments."
                            
                            logger.warning(f"❌ User rejected invoice: {basename}")
                            if feedback.strip():
                                logger.info(f"📝 Rejection feedback: {feedback[:100]}...")
                            
                            # Store feedback
                            st.session_state.human_feedback[basename] = rejection_text
                            
                            # Remove from interrupted items BEFORE resuming
                            st.session_state.interrupted_items = [
                                item for item in st.session_state.interrupted_items 
                                if item["basename"] != basename
                            ]
                            
                            # Resume workflow with Command
                            with st.spinner("Resuming workflow..."):
                                try:
                                    reports_from_resume = []
                                    # Process streamed events from resumed workflow
                                    for event in resume_workflow(
                                        thread_id=st.session_state.thread_id,
                                        feedback=rejection_text,
                                        basename=basename
                                    ):
                                        # Check for new interrupts
                                        if "__interrupt__" in event:
                                            # Another interrupt occurred
                                            interrupt_data = event["__interrupt__"]
                                            if len(interrupt_data) > 0:
                                                review_data = interrupt_data[0].value
                                                new_basename = review_data.get("basename", "Unknown")
                                                st.session_state.interrupted_items.append({
                                                    "basename": new_basename,
                                                    "review_data": review_data,
                                                    "report": review_data.get("report", {})
                                                })
                                        
                                        # Collect reports from resumed workflow
                                        for step, output in event.items():
                                            if step.startswith("__"):
                                                continue
                                            if step == "report" and isinstance(output, dict):
                                                report = output.get("report", {})
                                                if report:
                                                    reports_from_resume.append(report)
                                    
                                    # Update reports in session state
                                    for report in reports_from_resume:
                                        r_basename = report.get("json", {}).get("basename")
                                        if r_basename:
                                            found = False
                                            for idx, existing_r in enumerate(st.session_state.reports):
                                                if isinstance(existing_r, dict) and existing_r.get("json", {}).get("basename") == r_basename:
                                                    st.session_state.reports[idx] = report
                                                    found = True
                                                    break
                                            if not found:
                                                st.session_state.reports.append(report)
                                    
                                    # Check if workflow is completely done
                                    if len(st.session_state.interrupted_items) == 0:
                                        st.session_state.is_processing = False
                                    
                                    st.error(f"❌ {basename} rejected and indexed.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error resuming workflow: {e}")
                
                elif recommendation == "Manual Review":
                    st.info("This invoice requires manual review but workflow hasn't paused yet.")
                else:
                    if recommendation == "Approve":
                        st.success("✅ Auto-approved - No manual review required")
                    elif recommendation == "Reject":
                        st.error("❌ Auto-rejected")
            
            st.markdown("---")
    
    # Summary statistics
    st.header("Summary")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    total = len(st.session_state.reports)
    approved = sum(1 for r in st.session_state.reports if r.get("recommendation") == "Approve")
    manual = sum(1 for r in st.session_state.reports if r.get("recommendation") == "Manual Review")
    feedback_count = len(st.session_state.human_feedback)
    
    with col_stat1:
        st.metric("Total Processed", total)
    with col_stat2:
        st.metric("Auto-Approved", approved)
    with col_stat3:
        st.metric("Needs Review", manual)
    with col_stat4:
        st.metric("Feedback Provided", feedback_count)
    
    # Show interrupted status
    if st.session_state.interrupted_items:
        st.warning(f"⏸️ {len(st.session_state.interrupted_items)} invoice(s) waiting for feedback to continue processing")
    
else:
    st.info("Click 'Start Processing' above to begin processing invoices.")

# Debug section
with st.expander("🔧 Debug Info"):
    st.write("Thread ID:", st.session_state.thread_id)
    st.write("Interrupted Items:", len(st.session_state.interrupted_items))
    st.write("Human Feedback:", st.session_state.human_feedback)