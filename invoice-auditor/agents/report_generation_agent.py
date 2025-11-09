"""
Report HTML Generation Agent - Uses LLM to generate human-readable HTML report.
"""
import os
from typing import Any, Dict
from dotenv import load_dotenv


def generate_html_report(basename: str, extraction: Dict[str, Any], system_json: Dict[str, Any]) -> str:
    """Generate an HTML report string using an LLM.

    Args:
        basename: File basename or invoice identifier for title
        extraction: Translated invoice JSON used to show key fields
        system_json: SystemReport JSON (overview, recommendation, discrepancies, issues, etc.)

    Returns:
        HTML string
    """
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore

    load_dotenv()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant that outputs clean, self-contained HTML for an invoice validation report. "
            "Return only valid HTML (no markdown). Include a simple CSS for readability. "
            "Do not embed external images or scripts.\n\n"
            "Style Requirements (keep simple, professional, accessible):\n"
            "- Use a neutral font stack (e.g., system-ui, Arial, sans-serif) with 14-16px base size.\n"
            "- Provide generous whitespace and a max-width container (900-1100px).\n"
            "- Use a subtle header bar/title with invoice identifier.\n"
            "- Color palette: dark text (#222-#333), light background (#f7f7f7), white cards (#fff).\n"
            "- Section headings with consistent spacing and a subtle bottom border.\n"
            "- Status/recommendation badge with simple background colors: Approve (green #d4edda/#155724), Manual Review (amber #fff3cd/#856404), Reject (red #f8d7da/#721c24).\n"
            "- Tables (if any) with collapsed borders, zebra rows, and padding.\n"
            "- Responsive: content should scale on mobile (container padding, readable text).\n"
            "- Avoid heavy decoration; prioritize clarity and readability."
        )),
        ("human", (
            "Create an HTML report. Keep it simple and readable.\n\n"
            "Title: Invoice Validation Report - {basename}\n\n"
            "Overall Section: Show recommendation clearly with a colored status badge (Approve/Manual Review/Reject).\n"
            "Overview Section: Use system_json.overview.\n"
            "Key Info Section: show invoice_number, invoice_date, vendor_name, total (with currency), po_number from extraction as a VERTICAL list (one item per line). DO NOT use multi-column grids; ensure each key/value is on its own line.\n"
            "Issues Section: bullet list of system_json.issues (or 'No issues found').\n"
            "Discrepancy Summary Section: bullet list of system_json.discrepancy_summary (or 'No discrepancies').\n"
            "Missing Fields Section: bullet list of system_json.missing_fields (or 'All fields present').\n"
            "Translation Section: Show system_json.translation_confidence as a percentage with simple styling.\n"
            "Data provided:\n"
            "- system_json: {system_json}\n"
            "- extraction: {extraction}\n\n"
            "Return only the HTML string."
        )),
    ])

    html = (prompt | llm).invoke({
        "basename": basename,
        "system_json": system_json,
        "extraction": extraction
    })

    # Some models return a Message object; normalize to string
    if hasattr(html, "content"):
        return str(getattr(html, "content"))
    return str(html)


__all__ = ["generate_html_report"]


