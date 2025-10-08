import re
from typing import Tuple, Optional, List, Dict
from dateutil import parser as dtparser

# ------------ Amount parsing config ------------
AMT = r"(?:₱|PHP|Php|php)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))"

# High-priority tokens that usually mean THE total to pay
TOTAL_KEYS = [
    "grand total", "total amount due", "amount due", "total due",
    "total amount", "total payable", "amount payable", "balance due",
    "balance", "total"
]

# Low-priority / must-ignore lines (payments/tenders/change)
LOW_PRIORITY_KEYS = [
    "cash", "cash tendered", "tendered", "payment", "paid", "change", "sukli"
]

SKIP_STORE = {"receipt","invoice","official","sales","or#","tin","vat","pos","cashier","terminal"}

_def_amt = re.compile(AMT)
_word = re.compile(r"[A-Za-z][A-Za-z\-&' ]{2,}")

# ------------ Date parsing config ------------
DATE_HINTS = ["date","txn date","transaction date","billing date","issued","due date","period","period covered","statement date"]
DATE_PATTERNS = [
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",                            # YYYY-MM-DD
    r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",                          # MM/DD/YYYY or DD/MM/YYYY
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})",  # 10 Sep 2025
]

# ------------ Helpers ------------
def _norm(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s or "").strip()

def _amounts_in_text(text: str) -> List[float]:
    vals: List[float] = []
    for m in _def_amt.finditer(text):
        try:
            vals.append(float(m.group(1).replace(",", "")))
        except Exception:
            pass
    return vals

def _is_low_priority_line(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in LOW_PRIORITY_KEYS)

# ------------ Total extraction (layout-aware then text-only) ------------
def extract_total_layout(words: List[Dict], full_text: str) -> Optional[float]:
    """
    Prefer numbers near TOTAL-like tokens using line indexing from pytesseract.
    words: [{text, conf, left, top, width, height, block_num, par_num, line_num}]
    """
    if not words:  # fallback to text-only
        return extract_total_textonly(full_text)

    # Build line map
    lines: Dict[tuple, List[Dict]] = {}
    for w in words:
        ln = (w.get("block_num"), w.get("par_num"), w.get("line_num"))
        lines.setdefault(ln, []).append(w)

    ordered = sorted(lines.keys())

    # Scan for total keywords and amounts on same/next line
    for idx, ln in enumerate(ordered):
        line_words = lines[ln]
        line_txt = " ".join(_norm(w.get("text", "")) for w in line_words if _norm(w.get("text", "")))
        if not line_txt:
            continue
        if _is_low_priority_line(line_txt):
            # Payment/cash/change lines are not real totals
            continue

        low = line_txt.lower()
        if any(k in low for k in TOTAL_KEYS):
            # same line first
            amts = _amounts_in_text(line_txt)
            if amts:
                return max(amts)  # if multiple on the line, take the largest

            # sometimes the value is on the next printed line
            if idx + 1 < len(ordered):
                nxt_ln = ordered[idx + 1]
                nxt_txt = " ".join(_norm(w.get("text", "")) for w in lines[nxt_ln] if _norm(w.get("text", "")))
                if nxt_txt and not _is_low_priority_line(nxt_txt):
                    amts2 = _amounts_in_text(nxt_txt)
                    if amts2:
                        return max(amts2)

    # Fallback: biggest plausible amount in entire text (ignore tiny values if a big one exists)
    return extract_total_textonly(full_text)

def extract_total_textonly(text: str) -> Optional[float]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) Prefer lines with true total keywords, skipping payment/change lines
    for i, line in enumerate(lines):
        if _is_low_priority_line(line):
            continue
        low = line.lower()
        if any(k in low for k in TOTAL_KEYS):
            m = _def_amt.search(line)
            if not m and i + 1 < len(lines) and not _is_low_priority_line(lines[i + 1]):
                m = _def_amt.search(lines[i + 1])
            if m:
                return float(m.group(1).replace(",", ""))

    # 2) Fallback: take the largest amount in the whole text (typical for itemized receipts)
    amounts = [float(m.group(1).replace(",", "")) for m in _def_amt.finditer(text)]
    if not amounts:
        return None
    mx = max(amounts)
    # If everything is small, still return the max; in PH receipts, totals are rarely < 50
    return mx

# ------------ Store extraction ------------
def extract_store(text: str) -> Optional[str]:
    # Prefer first few lines with “wordy” content, avoid boilerplate tokens
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:12]:
        cand = line.strip("-—:| ")
        if len(cand) < 3:
            continue
        if any(k in cand.lower() for k in SKIP_STORE):
            continue
        if _word.search(cand):
            # Fix common OCR confusions
            cand = cand.replace("|", "I").replace("0/", "Q")
            return _norm(cand)
    return None

# ------------ Date extraction ------------
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
    # global fallback
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            iso = _try_parse_date(m.group(1))
            if iso:
                return iso
    return None

# ------------ Public APIs ------------
def parse_fields_from_ocr(rec: dict) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Use layout-aware total extraction when word boxes are present.
    Expects rec like:
      {
        "text": "...",
        "words": [{text, conf, left, top, width, height, block_num, par_num, line_num}, ...],
        ...
      }
    """
    text = rec.get("text", "") or ""
    words = rec.get("words") or []
    # If you pass a pre-cleaned amount text, it will still fall back to text if empty
    amount_source = rec.get("amount_pass") or text

    store = extract_store(text)
    total = extract_total_layout(words, amount_source)
    date = extract_date(text)
    return store, total, date

# Backwards-compatible helper (used by existing code)
def parse_fields(ocr_text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    return extract_store(ocr_text), extract_total_textonly(ocr_text), extract_date(ocr_text)
