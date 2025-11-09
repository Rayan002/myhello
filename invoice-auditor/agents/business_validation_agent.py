import json
import urllib.error
import urllib.request
from typing import Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv


def _http_get_json(url: str, timeout: int = 5) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        return {"error": str(e), "success": False}


def _load_rules() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "rules.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_sku_master() -> Any:
    sku_path = Path(__file__).resolve().parents[1] / "configs" / "sku_master.json"
    if sku_path.exists():
        try:
            return json.loads(sku_path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _build_stage2_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate.from_template(
        (
            "\nYou are performing Stage 2 Invoice Business Validation.\n\n"
            "Input invoice JSON:\n\n{invoice_json}\n\n"
            "Required header fields: {required_header}\n\n"
            "Required line item fields: {required_line}\n\n"
            "Accepted currencies: {accepted_currencies}\n\n"
            "Given:\n\n"
            "ERP PO JSON: {mock_data}\n\n"
            "ERP vendor JSON: {mock_vendor}\n\n"
            "SKU master array: {sku_master_data}\n\n"
            "Tolerances: {tolerances}\n\n"
            "Perform business validation deterministically based ONLY on provided ERP data and tolerances (no assumptions):\n\n"
            "- Vendor consistency: if vendor_id from PO maps to vendor JSON, set vendor_flags.vendor_name_match = \"match\"; else \"mismatch\".\n\n"
            "- Line items must exist in PO by item_code (alias 'code' acceptable in invoice).\n\n"
            "- Quantity discrepancy: use tolerances.quantity_difference_percent (percent of PO qty). If |inv_qty - po_qty| > po_qty * tol/100, flag quantity_mismatch and include numeric details.\n\n"
            "- Unit price discrepancy: use tolerances.price_difference_percent. Compute diff_pct = |inv_price - po_price|/po_price*100 (0 if po_price=0). If diff_pct > tol, flag price_mismatch and include invoice_price, po_price, and difference_percent.\n\n"
            "- Currency per line: if both present and differ, flag currency_mismatch.\n\n"
            "- Arithmetic per line: if qty and unit_price numeric, computed = qty*unit_price (round 2 decimals). If total present and different after rounding, mark computed_total_status=\"mismatch\".\n\n"
            "- SKU master mapping: if provided, require item_code to exist in SKU list; allow simple description substring checks when available.\n\n"
            "- Report missing items (in PO not in invoice, and vice versa).\n\n"
            "Return ONLY strict valid JSON.\n\n"
            "Echo the original invoice object unchanged PLUS add one new top-level key \"business_validation\".\n\n"
            "\"business_validation\" must be:\n\n"
            "{{\n\n"
            "\"vendor_flags\": {{\"vendor_name_match\": \"match\"|\"mismatch\", \"currency_match\": \"match\"|\"mismatch\"}},\n\n"
            "\"line_item_flags\": [\n\n{line_item_schema}\n\n],\n\n"
            "\"missing_in_invoice\": [<item_code strings>],\n\n"
            "\"missing_in_po\": [<item_code strings>],\n\n"
            "\"discrepancy_summary\": [<up to 30 short strings or [\"None\"]]  \n\n"
            "}}\n\n"
            "Rules:\n\n"
            "- Do NOT alter existing keys or values.\n\n"
            "- NO commentary before or after JSON.\n\n"
            "- All keys double-quoted.\n\n"
            "- Null where data unavailable.\n"
        ).strip()
    )


def _validate_business_with_llm_echo(translated_invoice: Dict[str, Any], po_data: Dict[str, Any], vendor: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import JsonOutputParser
    load_dotenv()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=2)
    prompt = _build_stage2_prompt()
    parser = JsonOutputParser()
    line_item_schema = (
        "{\"index\": <int>, \"item_code status\": \"match\" | \"mismatch\", \"qty_status\": \"match\"|\"mismatch\"|\"insufficient_data\",\n"
        "\"unit_price_status\": \"match\"|\"mismatch\"|\"insufficient_data\",\n"
        "\"item_code_status\": \"match\"|\"mismatch\",\n"
        "\"currency_status\": \"match\"|\"mismatch\"|\"insufficient_data\",\n"
        "\"computed_total_status\": \"match\"|\"mismatch\"|\"insufficient_data\",\n"
        "\"pct_qty_diff\": <float|null>, \"pct_unit_price_diff\": <float|null>,\n"
        "\"flags\": [<string>], \"status\": \"ok\"|\"warning\"|\"error\"}"
    )
    return (prompt | llm | parser).invoke({
        "invoice_json": json.dumps(translated_invoice, ensure_ascii=False),
        "required_header": rules.get("required_fields", {}).get("header", []),
        "required_line": rules.get("required_fields", {}).get("line_item", []),
        "accepted_currencies": rules.get("accepted_currencies", []),
        "mock_data": json.dumps(po_data, ensure_ascii=False),
        "mock_vendor": json.dumps(vendor, ensure_ascii=False),
        "sku_master_data": json.dumps(_load_sku_master(), ensure_ascii=False),
        "tolerances": json.dumps(rules.get("tolerances", {"pct_diff": 0.05}), ensure_ascii=False),
        "line_item_schema": line_item_schema,
    })


def validate_business_rules_agent(translated_invoice: Dict[str, Any], erp_base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is required for business validation")

    po_number = translated_invoice.get("po_number") or translated_invoice.get("po_reference")
    po_data = _http_get_json(f"{erp_base_url}/po/{po_number}", timeout=3) if po_number else {"error": "PO not found"}
    rules = _load_rules() or {}

    vendor = {}
    if po_data and isinstance(po_data, dict) and po_data.get("vendor_id"):
        vendor = _http_get_json(f"{erp_base_url}/vendor/{po_data['vendor_id']}", timeout=3)

    return _validate_business_with_llm_echo(translated_invoice, po_data, vendor, rules)


__all__ = ["validate_business_rules_agent"]
