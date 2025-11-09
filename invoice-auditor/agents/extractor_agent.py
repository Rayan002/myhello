import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    item_code: str = Field(description="Item code (e.g., SKU-001)")
    description: str = Field(description="Item description")
    qty: float = Field(description="Quantity (number, not string)")
    unit_price: float = Field(description="Unit price (number, not string)")
    total: float = Field(description="Line item total")


class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = Field(description="Invoice number (e.g., INV-1001)")
    invoice_date: Optional[str] = Field(description="Invoice date in YYYY-MM-DD format")
    vendor_name: Optional[str] = Field(description="Vendor name")
    currency: Optional[str] = Field(description="Currency ISO code (e.g., USD, EUR)")
    subtotal: Optional[float] = Field(description="Subtotal amount (before tax)")
    tax: Optional[float] = Field(description="Tax amount (if mentioned in invoice)")
    total: Optional[float] = Field(description="Total amount (including tax)")
    po_number: Optional[str] = Field(description="PO Reference/PO Number (e.g., PO-1001)")
    line_items: List[LineItem] = Field(default_factory=list, description="List of line items")


def _text_from_path(path: Path) -> str:
    sfx = path.suffix.lower()
    
    if sfx == ".pdf":
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(path)) or ""
            if text.strip():
                return text.strip()
        except Exception:
            pass
        
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            parts = []
            for pg in getattr(reader, "pages", []):
                try:
                    parts.append(pg.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(p for p in parts if p).strip()
        except Exception:
            return ""
    
    elif sfx == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception:
            return ""
    
    elif sfx in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
            img = Image.open(str(path))
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""
    
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_chain():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Provide it in the Streamlit sidebar.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(InvoiceExtraction)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an invoice extraction expert. Extract all invoice data from the provided text.\n\n"
                "CRITICAL FIELD NAME RULES:\n"
                "- Line items MUST use 'item_code' (NOT 'code'), 'qty' (NOT 'quantity'), 'unit_price' (NOT 'unit')\n"
                "- Extract PO Reference/PO Number from invoice if present (look for 'PO Reference:', 'PO Number:', 'PO:', etc.)\n"
                "- Extract subtotal, tax, if mentioned in the invoice (look for 'Subtotal:', 'Tax:', etc.)\n"
                "- All numeric fields must be numbers, not strings\n"
                "- Extract all line items with ALL fields (item_code, description, qty, unit_price, total)\n"
                "- If a field is missing in the invoice, use null for that field, DO NOT use dummy data",
            ),
            (
                "human",
                "Extract invoice data from the following text:\n\n{invoice_text}",
            ),
        ]
    )

    return prompt | structured_llm


def _normalize_extraction_fields(extraction: Dict) -> Dict:
    if not isinstance(extraction, dict):
        return {}
    
    normalized = extraction.copy()
    
    if "line_items" in normalized:
        normalized_line_items = []
        for item in normalized["line_items"]:
            if not isinstance(item, dict):
                continue
            normalized_item = item.copy()
            if "quantity" in normalized_item and "qty" not in normalized_item:
                normalized_item["qty"] = normalized_item.pop("quantity")
            if "unit" in normalized_item and "unit_price" not in normalized_item:
                normalized_item["unit_price"] = normalized_item.pop("unit")
            if "code" in normalized_item and "item_code" not in normalized_item:
                normalized_item["item_code"] = normalized_item.pop("code")
            
            normalized_line_items.append(normalized_item)
        normalized["line_items"] = normalized_line_items
    
    if "po_number" not in normalized:
        for po_field in ["po", "po_reference", "purchase_order"]:
            if po_field in normalized:
                normalized["po_number"] = normalized.pop(po_field)
                break
    
    if "total" in normalized:
        try:
            normalized["total"] = float(normalized["total"])
        except (ValueError, TypeError):
            pass
    
    for item in normalized.get("line_items", []):
        for field in ["total", "qty", "unit_price"]:
            if field in item:
                try:
                    item[field] = float(item[field])
                except (ValueError, TypeError):
                    pass
    
    return normalized


def extract_invoices_with_llm(monitor_results: List[Dict]) -> List[Dict]:
    if not monitor_results:
        return []

    chain = _build_chain()
    outputs = []
    
    for item in monitor_results:
        file_path = item.get("file_path")
        metadata_path = item.get("metadata_path")
        
        txt = ""
        if file_path:
            txt = _text_from_path(Path(file_path)).strip()
        
        metadata = {}
        if metadata_path:
            try:
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        try:
            result = chain.invoke({"invoice_text": txt})
            parsed = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        except Exception:
            parsed = {}
        
        parsed = _normalize_extraction_fields(parsed)

        outputs.append(
            {
                "basename": item.get("basename"),
                "file_path": item.get("file_path"),
                "metadata_path": item.get("metadata_path"),
                "file_type": item.get("file_type"),
                "metadata": metadata,
                "extraction": parsed,
            }
        )

    return outputs


__all__ = ["extract_invoices_with_llm"]
