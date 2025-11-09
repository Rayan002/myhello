import json
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    item_code: Optional[str] = Field(description="Item code (e.g., SKU-001)")
    description: Optional[str] = Field(description="Item description (translated to English)")
    qty: Optional[float] = Field(description="Quantity (number, unchanged)")
    unit_price: Optional[float] = Field(description="Unit price (number, unchanged)")
    total: Optional[float] = Field(description="Line item total (number, unchanged)")


class TranslatedInvoice(BaseModel):
    invoice_number: Optional[str] = Field(description="Invoice number (unchanged, e.g., INV-1001)")
    invoice_date: Optional[str] = Field(description="Invoice date in YYYY-MM-DD format (unchanged)")
    vendor_name: Optional[str] = Field(description="Vendor name (translated to English)")
    currency: Optional[str] = Field(description="Currency ISO code (unchanged, e.g., USD, EUR)")
    subtotal: Optional[float] = Field(description="Subtotal amount (unchanged)")
    tax: Optional[float] = Field(description="Tax amount (unchanged)")
    total: Optional[float] = Field(description="Total amount (unchanged)")
    po_number: Optional[str] = Field(description="PO Reference/PO Number (unchanged, e.g., PO-1001)")
    line_items: List[LineItem] = Field(default_factory=list, description="List of line items with descriptions translated")


class TranslationResult(BaseModel):
    language: str = Field(description="Detected language code")
    confidence: float = Field(description="Translation confidence (0.0 to 1.0)", ge=0.0, le=1.0)
    translated_invoice: TranslatedInvoice = Field(description="Translated invoice data with same schema as extraction")


def _build_chain():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Provide it in the Streamlit sidebar.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(TranslationResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation expert. Translate invoice data to English while preserving the exact JSON structure. "
                "CRITICAL RULES:\n"
                "1. Keep ALL numbers EXACTLY the same (invoice_number, dates, quantities, prices, totals, amounts)\n"
                "2. Translate ONLY text fields: vendor_name, line_item descriptions\n"
                "3. Keep currency codes unchanged (USD, EUR, etc.)\n"
                "4. Keep item_code values unchanged\n"
                "5. Preserve the exact same JSON schema structure\n"
                "6. Return the same fields as input, just with text translated\n"
                "Return structured output with language code, confidence (0.0-1.0), and translated_invoice (same schema as input).",
            ),
            (
                "human",
                "Translate the following invoice JSON to English. Keep all numbers and codes unchanged, translate only text fields:\n\n{input_json}\n\n"
                "Return the same JSON structure with text fields translated to English.",
            ),
        ]
    )
    
    return prompt | structured_llm


def translate_extractions_with_llm(extracted: List[Dict]) -> List[Dict]:
    if not extracted:
        return []

    chain = _build_chain()
    output = []
    
    for item in extracted:
        content = item.get("extraction") or {}
        
        if not content:
            output.append({**item, "translation": {
                "language": "en",
                "confidence": 1.0,
                "translated_invoice": content,
            }})
            continue
        
        try:
            input_json = json.dumps(content, ensure_ascii=False, indent=2)
            result = chain.invoke({"input_json": input_json})
            parsed = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            
            translated_invoice = parsed.get("translated_invoice", {})
            if isinstance(translated_invoice, dict) and hasattr(translated_invoice, "model_dump"):
                translated_invoice = translated_invoice.model_dump()
            
            output.append({
                **item,
                "translation": {
                    "language": parsed.get("language", "en"),
                    "confidence": parsed.get("confidence", 0.5),
                    "translated_invoice": translated_invoice,
                    "text_en": json.dumps(translated_invoice, ensure_ascii=False, indent=2),
                }
            })
        except Exception as exc:
            output.append({
                **item,
                "translation": {
                    "language": "unknown",
                    "confidence": 0.0,
                    "translated_invoice": content,
                    "text_en": json.dumps(content, ensure_ascii=False, indent=2),
                    "error": str(exc),
                }
            })
        
    return output


__all__ = ["translate_extractions_with_llm"]
