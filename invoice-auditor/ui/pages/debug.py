"""
Debug Mode - Detailed workflow execution logs
For development and troubleshooting purposes.
"""
import os
import sys
from pathlib import Path
import streamlit as st
import json
import uuid
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph_workflows.workflow import run_workflow_with_checkpoints, resume_workflow
from utils.logger_config import logger

st.set_page_config(
    page_title="Debug Mode - AI Invoice Auditor",
    layout="wide"
)

st.title("Debug Mode")
st.warning("This page shows detailed execution logs for development purposes.")
st.markdown("---")

# Initialize session state
if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = {}
if "reports" not in st.session_state:
    st.session_state.reports = []
if "debug_thread_id" not in st.session_state:
    st.session_state.debug_thread_id = str(uuid.uuid4())
if "interrupted_items" not in st.session_state:
    st.session_state.interrupted_items = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Action button
run_stream_btn = st.button("Run With Streaming", type="primary", use_container_width=True)

st.markdown("---")

# Run workflow with streaming
if run_stream_btn:
    st.session_state.debug_thread_id = str(uuid.uuid4())  # New thread for each run
    st.header("Streaming Execution Logs")
    
    # Create status container
    status_container = st.empty()
    status_container.info("Starting workflow execution...")
    
    # Create main log container
    st.markdown("### Workflow Steps")
    log_container = st.container(border=True)
    
    reports_list = []
    step_count = 0
    
    try:
        for event in run_workflow_with_checkpoints(
            thread_id=st.session_state.debug_thread_id,
            human_feedback=st.session_state.human_feedback
        ):
            # Check for interrupt events
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]
                if len(interrupt_data) > 0:
                    review_data = interrupt_data[0].value
                    basename = review_data.get("basename", "Unknown")
                    
                    # Store interrupt in session state
                    interrupt_item = {
                        "basename": basename,
                        "review_data": review_data,
                        "report": review_data.get("report", {})
                    }
                    # Check if already in list to avoid duplicates
                    if not any(item["basename"] == basename for item in st.session_state.interrupted_items):
                        st.session_state.interrupted_items.append(interrupt_item)
                    
                    with log_container:
                        st.warning(f"⏸️ **INTERRUPT** - Workflow paused for human feedback on: `{basename}`")
                        with st.expander("Interrupt Details", expanded=True):
                            st.json(review_data)
                        st.markdown("---")
                    
                    status_container.warning(f"⏸️ Workflow paused: {basename} requires manual review")
                    logger.warning(f"⏸️ Debug workflow interrupted: {basename}")
                    st.session_state.is_processing = False
                    break
            
            # Process normal workflow events
            for step, output in event.items():
                if step.startswith("__"):
                    continue
                
                step_count += 1
                
                # Update status in real-time
                status_container.info(f"Current Step: **{step}** ({step_count} steps executed)")
                
                # Display step in log container
                with log_container:
                    st.markdown(f"#### Step {step_count}: `{step}`")
                    
                    # Show current item if available
                    if isinstance(output, dict):
                        current_item = output.get("current_item", {})
                        if current_item:
                            basename = current_item.get("basename", "Unknown")
                            st.caption(f"📄 Processing: **{basename}**")
                    
                    # Display step output in expandable section
                    with st.expander(f"View {step} output", expanded=False):
                        st.json(output)
                    
                    st.markdown("---")
                
                # Collect reports
                if step == "report":
                    if isinstance(output, dict):
                        rep = output.get("report", {})
                        if rep:
                            reports_list.append(rep)
                            logger.info(f"📊 Report collected in debug mode: {rep.get('json', {}).get('basename', 'Unknown')}")
                
                # Handle human feedback step
                if step == "human_feedback":
                    with log_container:
                        st.info("👤 Human feedback step - workflow will pause for user input")
        
        # Final status
        if reports_list:
            st.session_state.reports = reports_list
            if len(st.session_state.interrupted_items) == 0:
                status_container.success(f"✅ Streaming completed. Generated {len(reports_list)} report(s).")
                logger.info(f"✅ Debug workflow completed: {len(reports_list)} report(s) generated")
                st.session_state.is_processing = False
            else:
                status_container.warning(f"⏸️ Workflow paused. {len(reports_list)} report(s) generated so far.")
        else:
            if len(st.session_state.interrupted_items) == 0:
                status_container.info("ℹ️ Workflow completed (no reports generated or workflow was interrupted)")
                logger.info("ℹ️ Debug workflow completed with no reports")
                st.session_state.is_processing = False
        
    except Exception as e:
        logger.error(f"❌ Error during debug workflow streaming: {str(e)}", exc_info=True)
        status_container.error(f"❌ Error during streaming: {str(e)}")
        with log_container:
            import traceback
            st.error("Full traceback:")
            st.code(traceback.format_exc())
        st.session_state.is_processing = False

# Handle interrupted items - show feedback form
if st.session_state.interrupted_items:
    st.markdown("---")
    st.header("⏸️ Human Feedback Required")
    
    for idx, item in enumerate(st.session_state.interrupted_items):
        basename = item["basename"]
        review_data = item["review_data"]
        report = item.get("report", {})
        
        with st.expander(f"📄 Invoice: {basename}", expanded=True):
            st.subheader(f"Invoice: {basename}")
            
            # Show report summary
            if report:
                rec = report.get("recommendation", "Unknown")
                st.write(f"**Recommendation:** {rec}")
                
                with st.expander("View Full Report", expanded=False):
                    st.json(report)
            
            # Feedback input
            feedback_key = f"feedback_{basename}_{idx}"
            feedback = st.text_area(
                "Enter your feedback:",
                key=feedback_key,
                placeholder="e.g., [HUMAN APPROVED] Looks good or [HUMAN REJECTED] Issue found...",
                height=100
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Approve & Resume", key=f"approve_{basename}_{idx}", use_container_width=True, type="primary"):
                    approval_text = f"[HUMAN APPROVED] {feedback}" if feedback.strip() else "[HUMAN APPROVED] No additional comments."
                    
                    logger.info(f"✅ User approved invoice: {basename}")
                    if feedback.strip():
                        logger.info(f"📝 Approval feedback: {feedback[:100]}...")
                    
                    # Store feedback
                    st.session_state.human_feedback[basename] = approval_text
                    
                    # Remove from interrupted items
                    st.session_state.interrupted_items = [
                        itm for itm in st.session_state.interrupted_items 
                        if itm["basename"] != basename
                    ]
                    
                    # Resume workflow
                    with st.spinner("Resuming workflow..."):
                        try:
                            reports_from_resume = []
                            # Process streamed events from resumed workflow
                            for event in resume_workflow(
                                thread_id=st.session_state.debug_thread_id,
                                feedback=approval_text,
                                basename=basename
                            ):
                                # Check for new interrupts
                                if "__interrupt__" in event:
                                    interrupt_data = event["__interrupt__"]
                                    if len(interrupt_data) > 0:
                                        new_review_data = interrupt_data[0].value
                                        new_basename = new_review_data.get("basename", "Unknown")
                                        st.session_state.interrupted_items.append({
                                            "basename": new_basename,
                                            "review_data": new_review_data,
                                            "report": new_review_data.get("report", {})
                                        })
                                
                                # Collect reports from resumed workflow
                                for step, output in event.items():
                                    if step.startswith("__"):
                                        continue
                                    if step == "report" and isinstance(output, dict):
                                        rep = output.get("report", {})
                                        if rep:
                                            reports_from_resume.append(rep)
                            
                            # Update reports in session state
                            for rep in reports_from_resume:
                                r_basename = rep.get("json", {}).get("basename")
                                if r_basename:
                                    found = False
                                    for i, existing_r in enumerate(st.session_state.reports):
                                        if isinstance(existing_r, dict) and existing_r.get("json", {}).get("basename") == r_basename:
                                            st.session_state.reports[i] = rep
                                            found = True
                                            break
                                    if not found:
                                        st.session_state.reports.append(rep)
                            
                            st.success(f"✅ Workflow resumed! {basename} approved and indexed.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error resuming workflow: {e}")
                            logger.error(f"❌ Error resuming workflow: {str(e)}", exc_info=True)
            
            with col2:
                if st.button("❌ Reject & Resume", key=f"reject_{basename}_{idx}", use_container_width=True):
                    rejection_text = f"[HUMAN REJECTED] {feedback}" if feedback.strip() else "[HUMAN REJECTED] No additional comments."
                    
                    logger.warning(f"❌ User rejected invoice: {basename}")
                    if feedback.strip():
                        logger.info(f"📝 Rejection feedback: {feedback[:100]}...")
                    
                    # Store feedback
                    st.session_state.human_feedback[basename] = rejection_text
                    
                    # Remove from interrupted items
                    st.session_state.interrupted_items = [
                        itm for itm in st.session_state.interrupted_items 
                        if itm["basename"] != basename
                    ]
                    
                    # Resume workflow
                    with st.spinner("Resuming workflow..."):
                        try:
                            reports_from_resume = []
                            # Process streamed events from resumed workflow
                            for event in resume_workflow(
                                thread_id=st.session_state.debug_thread_id,
                                feedback=rejection_text,
                                basename=basename
                            ):
                                # Check for new interrupts
                                if "__interrupt__" in event:
                                    interrupt_data = event["__interrupt__"]
                                    if len(interrupt_data) > 0:
                                        new_review_data = interrupt_data[0].value
                                        new_basename = new_review_data.get("basename", "Unknown")
                                        st.session_state.interrupted_items.append({
                                            "basename": new_basename,
                                            "review_data": new_review_data,
                                            "report": new_review_data.get("report", {})
                                        })
                                
                                # Collect reports from resumed workflow
                                for step, output in event.items():
                                    if step.startswith("__"):
                                        continue
                                    if step == "report" and isinstance(output, dict):
                                        rep = output.get("report", {})
                                        if rep:
                                            reports_from_resume.append(rep)
                            
                            # Update reports in session state
                            for rep in reports_from_resume:
                                r_basename = rep.get("json", {}).get("basename")
                                if r_basename:
                                    found = False
                                    for i, existing_r in enumerate(st.session_state.reports):
                                        if isinstance(existing_r, dict) and existing_r.get("json", {}).get("basename") == r_basename:
                                            st.session_state.reports[i] = rep
                                            found = True
                                            break
                                    if not found:
                                        st.session_state.reports.append(rep)
                            
                            st.success(f"✅ Workflow resumed! {basename} rejected and indexed.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error resuming workflow: {e}")
                            logger.error(f"❌ Error resuming workflow: {str(e)}", exc_info=True)

# Display current reports (for debugging)
if st.session_state.reports:
    st.markdown("---")
    st.header("Current Reports in Session")
    
    for idx, rep in enumerate(st.session_state.reports, 1):
        with st.expander(f"Report {idx}", expanded=False):
            st.json(rep)
