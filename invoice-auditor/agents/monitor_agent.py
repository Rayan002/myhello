import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _incoming_dir() -> Path:
    return _project_root() / "data" / "incoming"


def _ledger_path() -> Path:
    return _incoming_dir() / "_monitor_ledger.json"


def _load_ledger(ledger_file: Path) -> Dict:
    if not ledger_file.exists():
        return {"seen_basenames": [], "last_scan": None}
    try:
        return json.loads(ledger_file.read_text(encoding="utf-8"))
    except Exception:
        return {"seen_basenames": [], "last_scan": None}


def _save_ledger(ledger_file: Path, seen_basenames: List[str]) -> None:
    payload = {
        "seen_basenames": sorted(set(seen_basenames)),
        "last_scan": datetime.now(timezone.utc).isoformat(),
    }
    ledger_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check_for_new_invoices(
    incoming_dir: Optional[Path] = None,
    ledger_file: Optional[Path] = None,
) -> List[Dict]:
    """
    Scan the incoming directory for new files with matching metadata.json.
    Supports: .pdf, .docx, .png, .jpg, .jpeg + .json pairs.
    Maintains a JSON ledger and returns list with basename, paths, metadata, and text.
    """
    inbox = incoming_dir or _incoming_dir()
    ledger = ledger_file or _ledger_path()

    inbox.mkdir(parents=True, exist_ok=True)

    state = _load_ledger(ledger)
    seen = set(state.get("seen_basenames", []))

    # Supported file extensions
    supported_extensions = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}

    # Collect basenames from all supported files and filter by presence of matching JSON
    basenames = []
    for entry in os.listdir(inbox):
        file_path = Path(entry)
        if file_path.suffix.lower() in supported_extensions:
            base = file_path.stem
            meta_path = inbox / f"{base}.json"
            if meta_path.exists():
                basenames.append(base)

    new_basenames = [b for b in basenames if b not in seen]

    results: List[Dict] = []
    for base in new_basenames:
        # Find the actual file (could be pdf, docx, or image)
        file_path = None
        for ext in supported_extensions:
            candidate = inbox / f"{base}{ext}"
            if candidate.exists():
                file_path = candidate
                break

        if not file_path:
            continue

        meta_path = inbox / f"{base}.json"

        # Determine file type
        file_type = file_path.suffix.lower().replace(".", "")

        # Only pass paths - extraction will happen in extractor agent
        results.append(
            {
                "basename": base,
                "file_path": str(file_path),
                "metadata_path": str(meta_path),
                "file_type": file_type,  # pdf, docx, png, jpg, etc.
            }
        )

    # Update ledger with all paired basenames (new and existing)
    updated_seen = sorted(set(seen).union(basenames))
    _save_ledger(ledger, updated_seen)

    return results


# Helper utilities for UI
def get_incoming_dir() -> Path:
    return _incoming_dir()


def peek_new_basenames() -> List[str]:
    inbox = _incoming_dir()
    ledger = _ledger_path()
    inbox.mkdir(parents=True, exist_ok=True)
    state = _load_ledger(ledger)
    seen = set(state.get("seen_basenames", []))
    supported_extensions = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
    basenames = []
    for entry in os.listdir(inbox):
        file_path = Path(entry)
        if file_path.suffix.lower() in supported_extensions:
            base = file_path.stem
            meta_path = inbox / f"{base}.json"
            if meta_path.exists():
                basenames.append(base)
    return [b for b in basenames if b not in seen]


def save_uploaded_files(files: List, errors: List[str]) -> List[str]:
    """Save uploaded files to incoming dir. Ensures metadata JSON exists for binary files.
    Returns list of saved basenames; appends errors for missing metadata.
    """
    inbox = _incoming_dir()
    inbox.mkdir(parents=True, exist_ok=True)
    # Collect by basename
    by_name = {}
    for f in files:
        name = getattr(f, "name", None) or ""
        by_name[name] = f
    # Determine basenames present
    binaries = []
    jsons = set()
    for name in by_name.keys():
        p = Path(name)
        if p.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}:
            binaries.append(p)
        elif p.suffix.lower() == ".json":
            jsons.add(p.stem)
    saved = []
    for p in binaries:
        base = p.stem
        if base not in jsons:
            errors.append(f"Missing metadata JSON for {p.name} (expected {base}.json)")
            continue
    # Save all provided files
    for name, f in by_name.items():
        target = inbox / name
        try:
            target.write_bytes(f.read())
        except Exception:
            errors.append(f"Failed to save {name}")
        else:
            if Path(name).suffix.lower() != ".json":
                saved.append(Path(name).stem)
    return saved


__all__ = [
    "check_for_new_invoices",
    "get_incoming_dir",
    "peek_new_basenames",
    "save_uploaded_files"
]

