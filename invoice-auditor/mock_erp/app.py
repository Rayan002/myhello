"""
Simple FastAPI mock ERP API for invoice validation.
Returns vendor and PO data from JSON files.
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock ERP API", version="1.0.0")

# Data directory (same as erp_mock_data/)
DATA_DIR = Path(__file__).parent.parent / "data" / "erp_mock_data"


def _load_json_file(filename: str) -> list:
    """Load a JSON file from the data directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return []
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/")
def root():
    return {"message": "Mock ERP API", "version": "1.0.0"}


@app.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    """Get vendor information by vendor_id."""
    vendors = _load_json_file("vendors.json")
    for vendor in vendors:
        if vendor.get("vendor_id") == vendor_id:
            return JSONResponse(content=vendor)
    raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")


@app.get("/po/{po_number}")
def get_po(po_number: str):
    """Get Purchase Order information by po_number."""
    pos = _load_json_file("PO_Records.json")
    for po in pos:
        if po.get("po_number") == po_number:
            return JSONResponse(content=po)
    raise HTTPException(status_code=404, detail=f"PO {po_number} not found")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="0.0.0.0", port=8000)

