import re
import statistics
from typing import Tuple, Optional, List, Dict
from dateutil import parser as dtparser

# ------------ Amount parsing config ------------
AMT = r"(?:₱|PHP|Php|php)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))"

# High-priority tokens that usually mean THE total to pay
TOTAL_KEYS = [
    "grand total", "total amount due", "amount due", "total due",
    "total amount", "total payable", "amount payable", "balance due",
    "balance", "total", "amount to pay", "net amount due", "net amount",
    "total sales", "amount you owe", "Total",
]

# Low-priority / must-ignore lines (payments/tenders/change)
LOW_PRIORITY_KEYS = [
    "cash", "cash tendered", "tendered", "payment", "paid", "change", "sukli"
]

SKIP_STORE = {"receipt","invoice","official","sales","or#","tin","vat","pos","cashier","terminal"}

_def_amt = re.compile(AMT)
_word = re.compile(r"[A-Za-z][A-Za-z\-&' ]{2,}")
_currency = re.compile(r"(?:php|₱|php\.|peso|amount:)", re.IGNORECASE)

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

def _line_currency_score(line: str) -> float:
    return 1.0 if _currency.search(line) else 0.0

def _line_totalish_score(line: str) -> float:
    low = line.lower()
    score = 0.0
    if any(k in low for k in TOTAL_KEYS):
        score += 4.0
    if any(k in low for k in ("due", "payable", "amount due", "amount payable", "pay")):
        score += 1.5
    return score

def _score_amount_candidate(idx: int, line: str, value: float, lines: List[str]) -> float:
    score = 0.0
    score += _line_totalish_score(line)
    score += _line_currency_score(line)

    # Look at neighbors for supporting hints.
    if idx + 1 < len(lines):
        score += 0.8 * _line_totalish_score(lines[idx + 1])
        score += 0.4 * _line_currency_score(lines[idx + 1])
    if idx > 0:
        score += 0.6 * _line_totalish_score(lines[idx - 1])
        score += 0.3 * _line_currency_score(lines[idx - 1])

    # Prefer amounts near the bottom of the receipt.
    if len(lines) > 0:
        rel_pos = idx / len(lines)
        score += max(0.0, 2.0 * (1.0 - rel_pos))  # bottom lines earn up to +2

    # Light preference for numerically larger totals without letting them dominate.
    score += min(value, 50000.0) / 20000.0

    if _is_low_priority_line(line):
        score -= 3.0
    return score

def _best_amount(lines: List[str]) -> Optional[float]:
    best_val = None
    best_score = float("-inf")
    for idx, line in enumerate(lines):
        if _is_low_priority_line(line):
            continue
        for match in _def_amt.finditer(line):
            try:
                val = float(match.group(1).replace(",", ""))
            except Exception:
                continue
            score = _score_amount_candidate(idx, line, val, lines)
            # Tie-breaker: prefer larger value when score equal.
            if score > best_score or (abs(score - best_score) < 1e-6 and (best_val is None or val > best_val)):
                best_score = score
                best_val = val
    return best_val

# ------------ Total extraction (layout-aware then text-only) ------------
def extract_total_layout(words: List[Dict], full_text: str) -> Optional[float]:
    """
    Prefer numbers near TOTAL-like tokens using bounding boxes from pytesseract output.
    """
    if not words:
        return extract_total_textonly(full_text)

    tokens = []
    for w in words:
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        try:
            left = int(w.get("left", 0))
            top = int(w.get("top", 0))
            width = max(int(w.get("width", 0)), 1)
            height = max(int(w.get("height", 0)), 1)
            conf = float(w.get("conf", 0.0))
            if conf != conf:  # NaN guard
                conf = 0.0
            token = {
                "text": text,
                "conf": conf,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "block_num": int(w.get("block_num", 0)),
                "par_num": int(w.get("par_num", 0)),
                "line_num": int(w.get("line_num", 0)),
                "word_num": int(w.get("word_num", 0)),
            }
            token["right"] = token["left"] + token["width"]
            token["bottom"] = token["top"] + token["height"]
            token["center_x"] = token["left"] + token["width"] / 2.0
            token["center_y"] = token["top"] + token["height"] / 2.0
            tokens.append(token)
        except Exception:
            continue

    if not tokens:
        return extract_total_textonly(full_text)

    line_map: Dict[tuple, List[Dict]] = {}
    for tok in tokens:
        key = (tok["block_num"], tok["par_num"], tok["line_num"])
        line_map.setdefault(key, []).append(tok)

    ordered_keys = sorted(
        line_map.keys(),
        key=lambda k: (
            min(t["top"] for t in line_map[k]),
            min(t["left"] for t in line_map[k]),
        ),
    )

    line_infos: List[Dict] = []
    all_heights: List[int] = []

    for idx, key in enumerate(ordered_keys):
        line_tokens = sorted(line_map[key], key=lambda t: t["left"])
        if not line_tokens:
            continue
        line_text = " ".join(_norm(t["text"]) for t in line_tokens if _norm(t["text"]))
        x_min = min(t["left"] for t in line_tokens)
        x_max = max(t["right"] for t in line_tokens)
        y_top = min(t["top"] for t in line_tokens)
        y_bottom = max(t["bottom"] for t in line_tokens)
        all_heights.extend(t["height"] for t in line_tokens if t["height"] > 0)
        total_tokens = [
            t for t in line_tokens if any(k in t["text"].lower() for k in TOTAL_KEYS)
        ]
        line_infos.append({
            "index": idx,
            "text": line_text,
            "tokens": line_tokens,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "y_mid": (y_top + y_bottom) / 2.0,
            "x_mid": (x_min + x_max) / 2.0,
            "height_mean": sum(t["height"] for t in line_tokens) / len(line_tokens),
            "total_centers": [t["center_x"] for t in total_tokens],
        })

    if not line_infos:
        return extract_total_textonly(full_text)

    median_height = statistics.median(all_heights) if all_heights else 1.0
    median_height = median_height or 1.0
    line_count = len(line_infos)

    tot_lines = [
        info for info in line_infos if any(k in info["text"].lower() for k in TOTAL_KEYS)
    ]

    def _tokens_for_amount(line_tokens: List[Dict], match_str: str) -> List[Dict]:
        target = re.sub(r"[^0-9.,]", "", match_str)
        if not target:
            return [tok for tok in line_tokens if any(ch.isdigit() for ch in tok["text"])]
        acc = ""
        selected: List[Dict] = []
        for tok in line_tokens:
            cleaned = re.sub(r"[^0-9.,]", "", tok["text"])
            if not cleaned:
                continue
            candidate = acc + cleaned
            if target.startswith(candidate):
                selected.append(tok)
                acc = candidate
                if acc == target:
                    break
        if not selected:
            selected = [tok for tok in line_tokens if any(ch.isdigit() for ch in tok["text"])]
        return selected

    candidates = []
    for info in line_infos:
        if not info["text"]:
            continue
        for match in _def_amt.finditer(info["text"]):
            try:
                value = float(match.group(1).replace(",", ""))
            except Exception:
                continue
            subset = _tokens_for_amount(info["tokens"], match.group(1))
            if not subset:
                continue
            left = min(t["left"] for t in subset)
            right = max(t["right"] for t in subset)
            top = min(t["top"] for t in subset)
            bottom = max(t["bottom"] for t in subset)
            conf_vals = [t["conf"] for t in subset if t["conf"] == t["conf"]]
            avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
            avg_height = sum(t["height"] for t in subset) / len(subset)
            candidates.append({
                "value": value,
                "line": info,
                "tokens": subset,
                "bbox": (left, top, right, bottom),
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "avg_height": avg_height,
                "conf": avg_conf,
            })

    if not candidates:
        return extract_total_textonly(full_text)

    def _candidate_score(cand: Dict) -> float:
        line = cand["line"]
        idx = line["index"]
        score = 0.0
        score += _line_totalish_score(line["text"]) * 1.2
        score += _line_currency_score(line["text"])

        conf = cand.get("conf", 0.0)
        if conf == conf:
            score += min(conf, 95.0) / 25.0

        height_ratio = cand.get("avg_height", median_height) / (median_height or 1.0)
        if height_ratio > 1.1:
            score += min(height_ratio - 1.0, 2.5)

        if line_count > 1:
            rel_pos = idx / (line_count - 1)
        else:
            rel_pos = 1.0
        score += max(0.0, 2.2 * (1.0 - rel_pos))

        if tot_lines:
            best_prox = -5.0
            for tot_line in tot_lines:
                diff_idx = abs(idx - tot_line["index"])
                base = 0.0
                if diff_idx == 0:
                    base = 6.0
                elif diff_idx == 1:
                    base = 4.5
                elif diff_idx == 2:
                    base = 3.0
                else:
                    base = max(0.0, 3.0 - 0.7 * diff_idx)
                vert_gap = abs(cand["center_y"] - tot_line["y_mid"]) / (median_height or 1.0)
                base -= min(vert_gap, 4.0)
                centers = tot_line["total_centers"] or [tot_line["x_mid"]]
                horiz_gap = min(abs(cand["center_x"] - cx) for cx in centers)
                denom = max(cand["bbox"][2] - cand["bbox"][0], median_height)
                base -= min(horiz_gap / denom, 4.0)
                best_prox = max(best_prox, base)
            score += best_prox

        if _is_low_priority_line(line["text"]):
            score -= 4.0

        return score

    best_candidate = max(candidates, key=_candidate_score, default=None)
    if best_candidate is not None:
        return best_candidate["value"]

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
    best = _best_amount(lines)
    if best is None:
        return None
    return best

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
