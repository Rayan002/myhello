import json
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv


def _load_rules() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "rules.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _build_stage1_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate.from_template(
        (
            "\n   You are performing Stage 1 Invoice Data Validation.\n\n"
            "   Input invoice JSON:\n\n{invoice_json}\n\n"
            "   Required header fields: {required_header}\n\n"
            "   Required line item fields: {required_line}\n\n"
            "   Accepted currencies (ISO codes): {accepted_currencies}\n\n"
            "   Currency symbol map (normalize to ISO): {currency_symbol_map}\n\n"
            "   Field mapping rules: Treat header fields equivalently using these examples:\n"
            "   - invoice_no ↔ invoice_number\n"
            "   - vendor_id ↔ vendor_name (presence of either counts present, do not mark missing if one exists)\n"
            "   - total_amount ↔ total\n"
            "   - quantity ↔ qty\n"
            "   - unit_price ↔ unit\n\n"
            "   Produce STRICT JSON identical to the input plus one new top-level key `validation_remarks`.\n"
            "   Do NOT alter existing values or remove fields. Only append.\n\n"
            "   Define the new object as:\n"
            "   \"validation_remarks\": {{\n"
            "     \"header_missing_fields\": [<str>],\n"
            "     \"line_item_missing_fields\": [{{\"index\": <int>, \"missing\": [<str>]}}],\n"
            "     \"currency_status\": \"accepted\" | \"unaccepted\" | \"missing\",\n"
            "     \"sum_line_totals\": <float>,\n"
            "     \"tax\" : <float or null>\n"
            "     \"header_total_amount\": <float or null>,\n"
            "     \"total_difference\": <float>,\n"
            "     \"line_arithmetic_flags\": [{{\"index\": <int>, \"qty\": <float or null>, \"unit_price\": <float or null>, \"total\": <float or null>, \"computed\": <float or null>, \"status\": \"match\"|\"mismatch\"|\"insufficient_data\"}}],\n"
            "     \"basic_notes\": [<short strings, max 8>]\n"
            "   }}\n\n"
            "   Rules:\n"
            "   - A header field is missing only if ALL mapped equivalents are null/empty (e.g., invoice_no and invoice_number both missing).\n"
            "   - For line items, allow field aliases (quantity|qty, unit_price|unit, item_code|code).\n"
            "   - If header.total_amount is missing, set header_total_amount null and total_difference null.\n"
            "   - sum_line_totals: treat non-numeric line total as 0 and include in basic_notes.\n"
            "   - line arithmetic: if qty and unit_price numeric, computed = qty*unit_price (round 2 decimals). Compare to line total if present.\n"
            "   - Tax handling: If tax present, use it. Else if tax_rate present, tax = subtotal * tax_rate/100. Else default tax = subtotal * 0.10.\n"
            "   - Compute expected_total = subtotal + tax. Provide total_difference = |expected_total - header_total_amount| (0 if header missing).\n"
            "   - currency_status: Normalize header.currency using currency_symbol_map, then: \"missing\" if empty; \"accepted\" if in accepted list; else \"unaccepted\".\n"
            "   - Preserve ordering of existing top-level keys.\n"
            "   - Output ONLY JSON (no commentary).\n"
        ).strip()
    )


def _validate_with_llm_echo(translated_invoice: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import JsonOutputParser
    load_dotenv()
    
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=2)
    prompt = _build_stage1_prompt()
    parser = JsonOutputParser()
    return (prompt | llm | parser).invoke({
        "invoice_json": json.dumps(translated_invoice, ensure_ascii=False),
        "required_header": rules.get("required_fields", {}).get("header", []),
        "required_line": rules.get("required_fields", {}).get("line_item", []),
        "accepted_currencies": rules.get("accepted_currencies", []),
        "currency_symbol_map": rules.get("currency_symbol_map", {}),
    })


def validate_invoice_data_agent(translated_invoice: Dict[str, Any]) -> Dict[str, Any]:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is required for invoice data validation")
    rules = _load_rules() or {}
    return _validate_with_llm_echo(translated_invoice, rules)


__all__ = ["validate_invoice_data_agent"]
