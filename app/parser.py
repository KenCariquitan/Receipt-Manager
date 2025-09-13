import re
from typing import Tuple, Optional, List, Dict
from dateutil import parser as dtparser

AMT = r"(?:₱|PHP|Php|php)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))"
TOTAL_KEYS = [
    "grand total","amount due","amount payable","total amount","total",
    "cash tendered","balance due","balance"
]
NEG_KEYS = ["change","sukli"]
SKIP_STORE = {"receipt","invoice","official","sales","or#","tin","vat","pos","cashier","terminal"}

_def_amt = re.compile(AMT)
_word = re.compile(r"[A-Za-z][A-Za-z\-&' ]{2,}")
DATE_HINTS = ["date","txn date","transaction date","billing date","issued","due date","period","period covered","statement date"]
DATE_PATTERNS = [
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})",
]

def _norm(s: str) -> str: return re.sub(r"\s{2,}", " ", s or "").strip()

def _amounts_in_text(text: str) -> List[float]:
    vals = []
    for m in _def_amt.finditer(text):
        try: vals.append(float(m.group(1).replace(",", "")))
        except: pass
    return vals

def extract_total_layout(words: List[Dict], full_text: str) -> Optional[float]:
    """
    Prefer numbers near TOTAL-like tokens using line indexing from pytesseract.
    words: [{text, conf, left, top, width, height, block_num, par_num, line_num}]
    """
    if not words:  # fallback to text-only
        return extract_total_textonly(full_text)

    # Build line map
    lines = {}
    for w in words:
        ln = (w.get("block_num"), w.get("par_num"), w.get("line_num"))
        lines.setdefault(ln, []).append(w)

    # Scan for total keywords and amounts on same/next line
    for ln in sorted(lines.keys()):
        line_words = lines[ln]
        line_txt = " ".join(_norm(w["text"]) for w in line_words if _norm(w["text"]))
        low = line_txt.lower()
        if any(k in low for k in NEG_KEYS):
            continue
        if any(k in low for k in TOTAL_KEYS):
            # same line
            amts = _amounts_in_text(line_txt)
            if amts: return max(amts)
            # next line
            # find the next existing line key
            nxt = None
            for ln2 in sorted(lines.keys()):
                if ln2 > ln:
                    nxt = ln2; break
            if nxt:
                nxt_txt = " ".join(_norm(w["text"]) for w in lines[nxt] if _norm(w["text"]))
                amts2 = _amounts_in_text(nxt_txt)
                if amts2: return max(amts2)

    # Fallback: biggest plausible amount in entire text (ignore tiny values if a big one exists)
    return extract_total_textonly(full_text)

def extract_total_textonly(text: str) -> Optional[float]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Prefer TOTAL-like lines
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in NEG_KEYS): continue
        if any(k in low for k in TOTAL_KEYS):
            m = _def_amt.search(line) or ( _def_amt.search(lines[i+1]) if i+1 < len(lines) else None )
            if m: return float(m.group(1).replace(",", ""))
    # Else take largest
    amounts = [float(m.group(1).replace(",", "")) for m in _def_amt.finditer(text)]
    if not amounts: return None
    mx = max(amounts)
    return mx if mx > 50 else mx

def extract_store(text: str) -> Optional[str]:
    # Prefer first few lines with “wordy” content, avoid boilerplate tokens
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:12]:
        cand = line.strip("-—:| ")
        if len(cand) < 3: continue
        if any(k in cand.lower() for k in SKIP_STORE): continue
        if _word.search(cand):
            # Fix common OCR confusions
            cand = cand.replace("|", "I").replace("0/", "Q")
            return _norm(cand)
    return None

def _try_parse_date(s: str) -> Optional[str]:
    try:
        dt = dtparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None

def extract_date(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        low = line.lower()
        if any(h in low for h in DATE_HINTS):
            for look in (line, lines[i+1] if i+1 < len(lines) else ""):
                for pat in DATE_PATTERNS:
                    m = re.search(pat, look, re.IGNORECASE)
                    if m:
                        iso = _try_parse_date(m.group(1))
                        if iso: return iso
    # global fallback
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            iso = _try_parse_date(m.group(1))
            if iso: return iso
    return None

def parse_fields_from_ocr(rec: dict) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Use layout-aware total extraction when word boxes are present."""
    store = extract_store(rec.get("text", ""))
    total = extract_total_layout(rec.get("words") or [], rec.get("amount_pass") or rec.get("text",""))
    date = extract_date(rec.get("text", ""))
    return store, total, date

# Backwards-compatible helper (used by existing code)
def parse_fields(ocr_text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    return extract_store(ocr_text), extract_total_textonly(ocr_text), extract_date(ocr_text)
