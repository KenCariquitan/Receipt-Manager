import re
from typing import Tuple, Optional
from dateutil import parser as dtparser

AMT = r"(?:₱|PHP|Php|php)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))"
TOTAL_KEYS = [
    "grand total", "amount due", "amount payable", "total amount", "total",
    "cash tendered", "subtotal", "balance due", "balance", "amount due"
]
NEG_KEYS = ["change", "sukli"]
SKIP_STORE = {"receipt", "invoice", "official", "sales", "or#", "tin", "vat", "pos"}

_def_amt = re.compile(AMT)
_word = re.compile(r"[A-Za-z][A-Za-z\-&' ]{2,}")

DATE_HINTS = [
    "date", "txn date", "transaction date", "billing date", "issued", "due date",
    "period", "period covered", "statement date"
]
DATE_PATTERNS = [
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})",
]

def extract_total(text: str) -> Optional[float]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in NEG_KEYS):
            continue
        if any(k in low for k in TOTAL_KEYS):
            m = _def_amt.search(line)
            if not m and i+1 < len(lines):
                m = _def_amt.search(lines[i+1])
            if m:
                return float(m.group(1).replace(",", ""))
    amounts = [float(m.group(1).replace(",", "")) for m in _def_amt.finditer(text)]
    if not amounts:
        return None
    mx = max(amounts)
    if mx > 50:
        return mx
    return mx

def extract_store(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines()]
    for line in lines[:10]:
        cand = line.strip("-—:| ")
        if len(cand) < 3:
            continue
        if any(k in cand.lower() for k in SKIP_STORE):
            continue
        if _word.search(cand):
            return re.sub(r"\s{2,}", " ", cand)
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
                        if iso:
                            return iso
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            iso = _try_parse_date(m.group(1))
            if iso:
                return iso
    return None

def parse_fields(ocr_text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    return extract_store(ocr_text), extract_total(ocr_text), extract_date(ocr_text)
