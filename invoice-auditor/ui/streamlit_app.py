"""
AI Invoice Auditor - Main Dashboard
Main entry point for the invoice auditing application.
"""
import os
import sys
import json
from pathlib import Path
import streamlit as st

# Ensure project root on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Initialize session state
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}

# Page configuration
st.set_page_config(
    page_title="AI Invoice Auditor",
    layout="wide",
    initial_sidebar_state="expanded"
)

page = "Dashboard"

# Route to appropriate page
if page == "Dashboard":
    # Main Dashboard Content
    st.title("Dashboard")
    st.markdown("Welcome to the AI Invoice Auditor system!")
    st.markdown("---")
    
    # Load reports from outputs folder
    reports_dir = Path(__file__).resolve().parents[1] / "outputs" / "reports"
    all_reports = []
    if reports_dir.exists():
        for json_file in reports_dir.glob("*_report.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                all_reports.append({
                    "recommendation": json_data.get("recommendation", "Unknown")
                })
            except Exception:
                continue
    
    # Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_reports = len(all_reports)
    with col1:
        st.metric("Total Reports", total_reports)
    
    approved_count = sum(1 for r in all_reports if r.get("recommendation") == "Approve")
    with col2:
        st.metric("Approved", approved_count)
    
    manual_review_count = sum(1 for r in all_reports if r.get("recommendation") == "Manual Review")
    with col3:
        st.metric("Manual Review", manual_review_count)
    
    rejected_count = sum(1 for r in all_reports if r.get("recommendation") == "Reject")
    with col4:
        st.metric("Rejected", rejected_count)

# Note: For other pages, Streamlit's built-in multi-page system will handle navigation


