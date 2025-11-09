"""
View Reports Page - Browse and view all generated validation reports
"""
import os
import sys
import json
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

st.set_page_config(
    page_title="View Reports - AI Invoice Auditor",
    layout="wide"
)

st.title("View Reports")
st.markdown("Browse and review all generated invoice validation reports.")
st.markdown("---")

# Initialize session state for feedback only
if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = {}

# Load reports from outputs folder
def load_reports_from_folder():
    """Load all reports from outputs/reports folder."""
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "outputs" / "reports"
    
    if not reports_dir.exists():
        return []
    
    reports = []
    # Find all JSON report files
    for json_file in reports_dir.glob("*_report.json"):
        try:
            # Load JSON
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            # Find corresponding HTML file
            html_file = json_file.with_suffix(".html")
            html_content = ""
            if html_file.exists():
                html_content = html_file.read_text(encoding="utf-8")
            
            # Create report structure
            reports.append({
                "json": json_data,
                "report": html_content,
                "recommendation": json_data.get("recommendation", "Unknown"),
                "html_path": str(html_file),
                "json_path": str(json_file),
            })
        except Exception as e:
            continue
    
    # Sort by basename for consistent ordering
    reports.sort(key=lambda x: x.get("json", {}).get("basename", ""))
    return reports

# Load reports
all_reports = load_reports_from_folder()

# Main content
if not all_reports:
    st.info("No reports available. Navigate to Process Invoices to generate reports.")
else:
    st.success(f"Showing {len(all_reports)} report(s)")
    st.markdown("---")
    
    # Display reports
    for idx, report in enumerate(all_reports, 1):
        summary = report.get("json", {})
        basename = summary.get("basename", f"Invoice {idx}")
        recommendation = report.get("recommendation", "Unknown")
        extracted_data = summary.get("extracted_data", {})
        invoice_number = extracted_data.get("invoice_number", summary.get("invoice_number", "N/A"))
        
        # Color coding
        if recommendation == "Approve":
            badge = "Approved"
        elif recommendation == "Reject":
            badge = "Rejected"
        else:
            badge = "Manual Review"
        
        # Report card
        with st.expander(f"{basename} - {badge} | Invoice: {invoice_number}", expanded=False):
            # Quick info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recommendation", recommendation)
            with col2:
                confidence = summary.get("translation_confidence", 0)
                st.metric("Translation Confidence", f"{confidence:.2%}")
            with col3:
                issues_count = len(summary.get("issues", []))
                st.metric("Issues Found", issues_count)
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Full Report", "Summary", "Details", "Feedback"])
            
            with tab1:
                # HTML Report
                html_report = report.get("report", "")
                if html_report:
                    try:
                        import streamlit.components.v1 as components
                        components.html(html_report, height=800, scrolling=True)
                    except Exception:
                        st.markdown(html_report, unsafe_allow_html=True)
                else:
                    st.info("No HTML report available")
                
                # Download links
                html_path = report.get("html_path", "")
                if html_path and Path(html_path).exists():
                    with open(html_path, "rb") as f:
                        st.download_button(
                            "Download HTML Report",
                            f.read(),
                            file_name=f"{basename}_report.html",
                            mime="text/html"
                        )
            
            with tab2:
                # Overview
                st.subheader("Overview")
                overview = summary.get("overview", "")
                if overview:
                    st.markdown(overview)
                else:
                    st.info("No overview available")
                
                st.markdown("---")
                
                # Recommendation
                st.subheader("Recommendation")
                rec = summary.get("recommendation", recommendation)
                if rec == "Approve":
                    st.success(f"✅ {rec}")
                elif rec == "Reject":
                    st.error(f"❌ {rec}")
                else:
                    st.warning(f"⚠️ {rec}")
                
                st.markdown("---")
                
                # Key Information
                st.subheader("Key Information")
                info_col1, info_col2 = st.columns(2)
                
                # Use extracted_data from JSON
                report_extraction = extracted_data
                
                with info_col1:
                    st.write(f"**Invoice Number:** {report_extraction.get('invoice_number', invoice_number)}")
                    st.write(f"**Invoice Date:** {report_extraction.get('invoice_date', 'N/A')}")
                    st.write(f"**Vendor:** {report_extraction.get('vendor_name', 'N/A')}")
                
                with info_col2:
                    st.write(f"**Total Amount:** {report_extraction.get('total', report_extraction.get('total_amount', 'N/A'))}")
                    st.write(f"**Currency:** {report_extraction.get('currency', 'N/A')}")
                    st.write(f"**PO Number:** {report_extraction.get('po_number', 'N/A')}")
                
                st.markdown("---")
                
                # Summary Details
                st.subheader("Summary Details")
                sum_col1, sum_col2 = st.columns(2)
                
                with sum_col1:
                    st.write("**Translation Confidence:**")
                    conf = summary.get("translation_confidence", 0)
                    st.metric("Translation Confidence", f"{conf:.2%}", label_visibility="hidden")
                    
                    st.write("**Issues Found:**")
                    issues = summary.get("issues", [])
                    if issues:
                        for issue in issues:
                            st.error(f"• {issue}")
                    else:
                        st.success("No issues found")
                
                with sum_col2:
                    st.write("**Discrepancies:**")
                    discrepancies = summary.get("discrepancy_summary", [])
                    if discrepancies:
                        for disc in discrepancies:
                            st.warning(f"• {disc}")
                    else:
                        st.success("No discrepancies")
                    
                    st.write("**Missing Fields:**")
                    missing_fields = summary.get("missing_fields", [])
                    if missing_fields:
                        for field in missing_fields:
                            st.warning(f"• {field}")
                    else:
                        st.success("All fields present")
            
            with tab3:
                # Detailed JSON
                st.subheader("Complete Report Data")
                st.json(summary)
                
                # Validation Details
                st.subheader("Validation Details")
                issues = summary.get("issues", [])
                discrepancies = summary.get("discrepancy_summary", [])
                missing_fields = summary.get("missing_fields", [])
                
                col_det1, col_det2 = st.columns(2)
                
                with col_det1:
                    st.write("Issues:")
                    if issues:
                        for issue in issues:
                            st.error(f"• {issue}")
                    else:
                        st.success("No issues")
                    
                    st.write("Missing Fields:")
                    if missing_fields:
                        for field in missing_fields:
                            st.warning(f"• {field}")
                    else:
                        st.success("All fields present")
                
                with col_det2:
                    st.write("Discrepancies:")
                    if discrepancies:
                        for disc in discrepancies:
                            st.warning(f"• {disc}")
                    else:
                        st.success("No discrepancies")
            
            with tab4:
                # Human Feedback
                st.subheader("Human Feedback")
                
                # Show existing feedback
                if basename in st.session_state.human_feedback:
                    st.success("Feedback Recorded:")
                    st.text(st.session_state.human_feedback[basename])
                else:
                    st.info("No feedback recorded yet")
                
                st.markdown("---")
                
                # Add/Edit feedback
                feedback = st.text_area(
                    "Add or update feedback",
                    value=st.session_state.human_feedback.get(basename, ""),
                    key=f"feedback_view_{basename}",
                    height=150,
                )
                
                if st.button("Save Feedback", key=f"save_view_{basename}"):
                    st.session_state.human_feedback[basename] = feedback
                    st.success("Feedback saved!")
                    st.rerun()
            
            st.markdown("---")
