"""
Reporting - LLM for system JSON and LLM for HTML generation (no Jinja, no PDF).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from agents.report_generation_agent import generate_html_report


class SystemReport(BaseModel):
    """JSON for system use (LLM-produced concise summary)."""
    basename: str = Field(description="File basename")
    overview: str = Field(description="Detailed Summary of the invoice and validation outcome")
    recommendation: str = Field(description="Approve / Manual Review / Reject")
    discrepancy_summary: List[str] = Field(default_factory=list, description="List of discrepancies")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")
    translation_confidence: float = Field(description="Translation confidence score (0..1)")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Extracted invoice data")


class FinalReport(BaseModel):
    """Final output structure with required three keys."""
    json: SystemReport = Field(description="JSON report for system use")
    report: str = Field(description="Human-readable report (HTML)")
    recommendation: str = Field(description="Approve / Manual Review / Reject")


def _load_rules() -> Dict[str, Any]:
    """Load validation rules from configs/rules.yaml."""
    path = Path(__file__).resolve().parents[1] / "configs" / "rules.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _project_reports_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "reports"


def generate_reports_with_llm(validated: List[Dict]) -> List[Dict]:
    """Generate system JSON via LLM and HTML via LLM agent. Save JSON+HTML under outputs/reports at project root."""
    if not validated:
        return []
    output: List[Dict] = []
    rules = _load_rules()

    # Compact summary chain
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Provide it in the Streamlit sidebar.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    summary_llm = llm.with_structured_output(SystemReport)
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You generate a concise JSON report for system use about an invoice validation. "
            "Strictly output JSON matching SystemReport with keys: basename, overview of invoice and validation outcome (5-6 sentences), "
            "recommendation (Approve/Manual Review/Reject based on policies), discrepancy_summary (list), missing_fields (list), "
            "translation_confidence (0..1), issues (list). No extra keys. No HTML.",
        ),
        (
            "human",
            "Analyze this invoice validation and return concise JSON with recommendation based on these policies.\n"
            "basename: {basename}\n"
            "issues: {issues}\n"
            "discrepancy_summary: {discrepancies}\n"
            "missing_fields: {missing_fields}\n"
            "translation_confidence: {translation_confidence}\n"
            "policies: {policies}\n"
            "Return JSON only.",
        ),
    ])
    summary_chain = summary_prompt | summary_llm

    for entry in validated:
        item = entry.get("item") or {}
        issues = entry.get("issues") or []
        business_validation = entry.get("business_validation", {})
        translation = item.get("translation") or {}
        translation_confidence = translation.get("confidence", 0.3)
        basename = item.get("basename") or (entry.get("extraction") or {}).get("invoice_number", "")
        extraction = entry.get("extraction", {})

        # Extract messages from Discrepancy objects
        discrepancies_msgs = []
        for d in business_validation.get("discrepancies", []):
            if isinstance(d, dict):
                msg = d.get("message", "")
                if msg:
                    discrepancies_msgs.append(msg)
            elif hasattr(d, "message"):  # Pydantic model
                if d.message:
                    discrepancies_msgs.append(d.message)
        policies = rules.get("validation_policies", {})
        missing_fields = (entry.get("data_validation") or {}).get("validation_details", {}).get("field_completeness", {}).get("missing_fields", [])

        try:
            sys_json_obj = summary_chain.invoke({
                "basename": basename,
                "issues": issues,
                "discrepancies": discrepancies_msgs,
                "missing_fields": missing_fields,
                "translation_confidence": round(float(translation_confidence), 3),
                "policies": json.dumps(policies, ensure_ascii=False),
            })
            sys_json = sys_json_obj.model_dump() if hasattr(sys_json_obj, "model_dump") else sys_json_obj
            # Add extracted data to the JSON
            sys_json["extracted_data"] = extraction
        except Exception as exc:
            # No fallback: return structured error entry
            output.append({
                "error": f"Reporting error: {str(exc)}",
                "input_basename": basename,
            })
            continue

        # HTML via generation agent
        recommendation = sys_json.get("recommendation", "Manual Review")
        html = generate_html_report(basename=basename, extraction=extraction, system_json=sys_json)

        # Save under project root
        output_dir = _project_reports_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        invoice_number = extraction.get("invoice_number", basename)
        # Sanitize invoice number for filename
        safe_invoice_num = "".join(c for c in str(invoice_number) if c.isalnum() or c in ("-", "_"))
        html_path = output_dir / f"{safe_invoice_num}_report.html"
        json_path = output_dir / f"{safe_invoice_num}_report.json"
        
        # Save HTML and JSON
        html_path.write_text(html, encoding="utf-8")
        try:
            json_path.write_text(json.dumps(sys_json, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        
        final_obj = FinalReport(json=SystemReport(**sys_json), report=html, recommendation=recommendation)
        result_dict = final_obj.model_dump()
        result_dict["html_path"] = str(html_path)
        result_dict["json_path"] = str(json_path)
        output.append(result_dict)
    return output


__all__ = ["generate_reports_with_llm"]
