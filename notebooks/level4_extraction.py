# level4_extraction.py
# ---------------------
from __future__ import annotations
import re
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# Optional dependencies handled gracefully
def _extract_text_pdfplumber(path: str) -> str:
    try:
        import pdfplumber
    except Exception:
        return ""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text_parts).strip()

def _extract_text_pypdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return ""
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    except Exception:
        return ""
    return "\n".join(text_parts).strip()

def extract_text_from_pdf(path: str) -> str:
    # Try multiple extractors; return the first with sufficient words
    for fn in (_extract_text_pdfplumber, _extract_text_pypdf):
        txt = fn(path)
        if txt and len(txt.split()) > 3:
            return txt
    return ""

# (Optional) OCR for scanned PDFs if pytesseract & pillow are installed
def ocr_pdf_first_n_pages(path: str, n_pages: int = 3, dpi: int = 300) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return ""
    text_parts = []
    try:
        images = convert_from_path(path, dpi=dpi, first_page=1, last_page=n_pages)
        for img in images:
            try:
                text_parts.append(pytesseract.image_to_string(img) or "")
            except Exception:
                continue
    except Exception:
        return ""
    return "\n".join(text_parts).strip()

# ---------------- Parsing ----------------
DATE_PATTERNS = [
    r"(\b\d{4}-\d{2}-\d{2}\b)",
    r"(\b\d{2}/\d{2}/\d{4}\b)",
    r"(\b\d{1,2}-\d{1,2}-\d{2,4}\b)",
]
TIME_PATTERNS = [r"(\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b)"]
PD_DISTRICT_PATTERNS = [r"\b(PD\s*District|Police\s*District|District)[:\-\s]+([A-Za-z\s]+)\b"]
ADDRESS_PATTERNS = [r"\bAddress[:\-\s]+(.+)$", r"\bLocation[:\-\s]+(.+)$"]
COORD_PATTERNS = [
    r"Latitude\s*\(?Y\)?[:\-\s]*(-?\d+\.\d+)\b.*Longitude\s*\(?X\)?[:\-\s]*(-?\d+\.\d+)\b",
    r"\bLat[:\-\s]*(-?\d+\.\d+)[,\s]+Lon[g]?[:\-\s]*(-?\d+\.\d+)\b",
]

def _first_group(patterns: List[str], text: str, flags=re.MULTILINE) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(m.lastindex or 0)
    return None

def parse_fields_from_text(text: str) -> Dict[str, Any]:
    out = {
        "Descript": text.strip(),
        "Dates": None,
        "Address": None,
        "PdDistrict": None,
        "Longitude (X)": None,
        "Latitude (Y)": None,
    }
    # Parse date and (optionally) time
    date_raw = _first_group(DATE_PATTERNS, text) or ""
    time_raw = _first_group(TIME_PATTERNS, text) or ""
    date_val = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m-%d-%y"):
        try:
            if date_raw:
                date_val = datetime.strptime(date_raw, fmt)
                break
        except ValueError:
            continue
    if date_val and time_raw:
        time_clean = time_raw.upper().replace(" ", "")
        for tfmt in ("%H:%M", "%H:%M:%S", "%I:%M%p", "%I:%M:%S%p"):
            try:
                t = datetime.strptime(time_clean, tfmt)
                date_val = date_val.replace(hour=t.hour, minute=t.minute, second=getattr(t, "second", 0))
                break
            except ValueError:
                continue
    out["Dates"] = date_val.isoformat(sep=" ") if date_val else None

    # Parse PD district
    pd_m = None
    for pat in PD_DISTRICT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            pd_m = m.group(m.lastindex or 0)
            break
    out["PdDistrict"] = pd_m.strip() if pd_m else None

    # Parse address (take first line only if multi-line)
    addr = None
    for pat in ADDRESS_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            addr = m.group(m.lastindex or 0)
            if "\n" in addr:
                addr = addr.split("\n", 1)[0]
            break
    out["Address"] = addr.strip() if addr else None

    # Parse coordinates if present
    for pat in COORD_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                lat = float(m.group(1)); lon = float(m.group(2))
                out["Latitude (Y)"] = float(lat)
                out["Longitude (X)"] = float(lon)
            except Exception:
                pass
            break
    return out

def features_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # Build a clean DataFrame and derive basic time features
    df = pd.DataFrame(rows)
    for col in ["Descript", "Address", "PdDistrict"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    if "Dates" in df.columns:
        df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
        df["Year"] = df["Dates"].dt.year
        df["Month"] = df["Dates"].dt.month
        df["DayOfMonth"] = df["Dates"].dt.day
        df["Hour"] = df["Dates"].dt.hour
        df["DayOfWeek"] = df["Dates"].dt.day_name()
    else:
        df["Dates"] = pd.NaT
        df["Year"] = None; df["Month"] = None; df["DayOfMonth"] = None; df["Hour"] = None; df["DayOfWeek"] = None

    for c in ["Latitude (Y)", "Longitude (X)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reorder columns for consistency
    preferred_cols = [
        "Dates","Descript","Address","PdDistrict",
        "Latitude (Y)","Longitude (X)",
        "Year","Month","DayOfMonth","Hour","DayOfWeek"
    ]
    preferred_in_df = [c for c in preferred_cols if c in df.columns]
    rest = [c for c in df.columns if c not in preferred_in_df]
    df = df[preferred_in_df + rest]
    return df