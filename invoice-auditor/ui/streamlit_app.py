import sys
import json
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}

st.set_page_config(
    page_title="AI Invoice Auditor",
    layout="wide",
    initial_sidebar_state="expanded"
)

if True:  # Dashboard page
    st.title("Dashboard")
    st.markdown("Welcome to the AI Invoice Auditor system!")
    st.markdown("---")
    
    reports_dir = Path(__file__).resolve().parents[1] / "outputs" / "reports"
    all_reports = []
    if reports_dir.exists():
        for json_file in reports_dir.glob("*_report.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                all_reports.append({"recommendation": json_data.get("recommendation", "Unknown")})
            except Exception:
                continue
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", len(all_reports))
    with col2:
        st.metric("Approved", sum(1 for r in all_reports if r.get("recommendation") == "Approve"))
    with col3:
        st.metric("Manual Review", sum(1 for r in all_reports if r.get("recommendation") == "Manual Review"))
    with col4:
        st.metric("Rejected", sum(1 for r in all_reports if r.get("recommendation") == "Reject"))


