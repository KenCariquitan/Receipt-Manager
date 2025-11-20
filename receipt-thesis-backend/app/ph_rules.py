from __future__ import annotations
import re
from typing import Optional, Tuple, Iterable
import difflib
from functools import lru_cache
import Levenshtein as lev  # pip install python-Levenshtein

# ================= Canonical brand sets (UPPERCASE) =================
FOOD_BRANDS = {
    "JOLLIBEE","MCDONALD","MCDONALD'S","KFC","CHOWKING","GREENWICH",
    "MANG INASAL","SHAKEY'S","BONCHON","STARBUCKS","GONG CHA","CHATIME",
    "7-ELEVEN", "MINISTOP", "FAMILYMART"
}
GROCERY_BRANDS = {
    "SM SUPERMARKET","SM HYPERMARKET","PUREGOLD","ROBINSONS SUPERMARKET",
    "WALTERMART","LANDERS","S&R", "GMALL","GRANDMALL"
}
UTILITY_BRANDS = {
    "MERALCO","PLDT","GLOBE","SMART","CONVERGE","MAYNILAD","MANILA WATER","SKY","DITO"
}
TRANSPORT_BRANDS = {
    "PETRON","SHELL","CALTEX","SEAOIL","EASYTRIP","AUTOSWEEP","GRAB","ANGKAS","NLEX","SLEX"
}
HEALTH_BRANDS = {
    "MERCURY DRUG","WATSONS","SOUTHSTAR","GENERIKA","ROSE PHARMACY","THE GENERICS PHARMACY"
}

# Brand aliases to handle OCR variants and legal names
ALIAS_MAP = {
    "GOLDEN ARCHES": "MCDONALD'S",
    "GOLDEN ARCHES FOOD CORPORATION": "MCDONALD'S",
    "GIANT ARCHES": "MCDONALD'S",
    "GIANT ARCHES FOOD CORPORATION": "MCDONALD'S",
    "WWW.UNIQLO.COM": "UNIQLO",
    "UNIQLO.COM": "UNIQLO",
    "BDA ENTERPRISES": "7-ELEVEN",
}

# Per-category keyword hints (lowercase)
UTILITY_KW = {"kwh","kilowatt","meter","account no","service period","due date","statement","internet","fiber","dsl","postpaid","prepaid load","load","data pack","billing"}
TRANSPORT_KW = {"diesel","unleaded","gasoline","pump","liter","litre","toll","rfid","easytrip","autosweep","plate","odometer","grab","angkas"}
FOOD_KW = {"meal","combo","burger","fries","chicken","rice","drink","beverage","snack","dine","take out"}
HEALTH_KW = {"pharmacy","rx","tablet","capsule","mg","ml","clinic","dental","optical","laboratory","prescription"}
GROCERY_KW = {"grocery","supermarket","hypermarket","market","minimart","convenience"}

ALL_SETS = [
    ("Utilities", UTILITY_BRANDS, UTILITY_KW),
    ("Transportation", TRANSPORT_BRANDS, TRANSPORT_KW),
    ("Health & Wellness", HEALTH_BRANDS, HEALTH_KW),
    ("Groceries", GROCERY_BRANDS, GROCERY_KW),
    ("Food", FOOD_BRANDS, FOOD_KW),
]

SPACES = re.compile(r"\s+")
BREAK_PAT = re.compile(
    r"\b(branch|tin|vat|address|add\.?|tel|contact|phone|no\.?|receipt|invoice|official|cashier|terminal|store no\.?)\b",
    re.IGNORECASE,
)

# ================= OCR-specific sanitize: fix common confusions =================
# Especially for 7-ELEVEN variants like "¢-ELEWEM", "¢-ELEWEOD"
def _sanitize_ocr(s: str) -> str:
    if not s:
        return s
    u = s.upper()

    # Replace odd glyphs often misread for '7' or '-'
    u = u.replace("¢", "7")  # OCR weirdness
    u = u.replace("€", "C")  # rare, but avoid harming ELEVEN
    u = u.replace("—", "-").replace("–", "-").replace("_", "-").replace("~", "-").replace("|", "I")
    u = u.replace("0/", "Q")  # store header pattern sometimes

    # Common ELEVEN misspellings from OCR
    u = u.replace("ELEWEM", "ELEVEN").replace("ELEWEOD", "ELEVEN").replace("ELEWEN", "ELEVEN").replace("ELEVENN", "ELEVEN")
    # Sometimes hyphen lost or repeated
    u = re.sub(r"\b7\s*ELEVEN\b", "7-ELEVEN", u)

    # Collapse spaces
    u = SPACES.sub(" ", u).strip()
    return u

_CHAR_REPLACEMENTS = (
    ("0", "O"),
    ("1", "I"),
    ("2", "Z"),
    ("3", "B"),
    ("4", "A"),
    ("5", "S"),
    ("6", "G"),
    ("7", "T"),
    ("8", "B"),
    ("9", "G"),
    ("@", "A"),
    ("$", "S"),
    ("€", "E"),
    ("£", "L"),
    ("¢", "C"),
)


def _apply_char_map(u: str) -> str:
    # Special-case some two-character confusions frequently seen in OCR
    u = u.replace("2I", "PI")
    u = u.replace("2L", "PL")
    for wrong, right in _CHAR_REPLACEMENTS:
        u = u.replace(wrong, right)
    return u


def _normalize_for_match(s: str) -> str:
    s = _sanitize_ocr(s)
    s = _apply_char_map(s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    # Trim leading non-alpha that might survive
    s = re.sub(r"^[^A-Z]+", "", s)
    return s

def normalize_store_name(store: Optional[str]) -> Optional[str]:
    if not store:
        return None
    s = _sanitize_ocr(store)
    # Only keep the first line if multi-line header came through
    if "\n" in s:
        s = s.split("\n", 1)[0]
    # Cut off at common keywords that usually start addresses or metadata
    match = BREAK_PAT.search(s)
    if match:
        s = s[:match.start()]
    # Remove legal suffixes
    s = re.sub(r"\b(CORP(?:ORATION)?|INC\.?|CO\.?|COMPANY|LTD\.?|CORPORATION)\b", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# ================= Fuzzy snapping to canonical brand =================
def _distance(a: str, b: str) -> int:
    return lev.distance(a, b)

def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _partial_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    span = len(a)
    for i in range(0, len(b) - span + 1):
        window = b[i : i + span]
        best = max(best, difflib.SequenceMatcher(None, a, window).ratio())
        if best >= 0.995:  # early exit if essentially perfect
            break
    return best


def _similarity_score(norm: str, candidate: str) -> float:
    """
    Combined score that considers raw and sanitized strings.
    Returns value in [0,1].
    """
    norm_clean = _normalize_for_match(norm)
    cand_clean = _normalize_for_match(candidate)

    if not norm_clean or not cand_clean:
        return 0.0

    lev_d = _distance(norm_clean, cand_clean)
    denom = max(len(norm_clean), len(cand_clean)) or 1
    lev_score = 1.0 - (lev_d / denom)

    seq_score = _sequence_ratio(norm_clean, cand_clean)
    partial = _partial_ratio(norm_clean, cand_clean)

    # Encourage prefix matches (common when OCR drops leading characters)
    prefix_bonus = 0.0
    if cand_clean.startswith(norm_clean) or norm_clean.startswith(cand_clean):
        prefix_bonus = 0.05

    base = max(lev_score, seq_score, partial)
    return min(1.0, base + prefix_bonus)


@lru_cache(maxsize=1)
def _all_brands_cached() -> tuple[str, ...]:
    brands: list[str] = []
    for _, bset, _ in ALL_SETS:
        brands.extend(list(bset))
    return tuple(brands)


def _best_match(norm: str, candidates: Iterable[str]) -> tuple[Optional[str], float]:
    """
    Return (best_brand, score) where score is normalized similarity in [0,1],
    using 1 - distance/len(max(a,b)).
    """
    if not norm:
        return None, 0.0
    best = None
    best_score = 0.0
    for c in candidates:
        score = _similarity_score(norm, c)
        if score > best_score:
            best, best_score = c, score
    return best, best_score

def correct_store_name(store: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Try to correct OCR'd store to a canonical brand.
    Returns (canonical_store, category, score) or (None, None, None) if not confident.
    """
    if not store:
        return None, None, None

    norm = normalize_store_name(store)
    if not norm:
        return None, None, None

    # Direct alias map check (handles legal names like "Golden Arches Food Corporation")
    for alias, canonical in ALIAS_MAP.items():
        if alias in norm.upper():
            for cat, brands, _ in ALL_SETS:
                if canonical in brands:
                    return canonical, cat, 1.0

    # 1) Exact/contains quick pass
    for cat, brands, _ in ALL_SETS:
        for b in brands:
            if b in norm or norm in b:
                return b, cat, 1.0

    # 2) Fuzzy against all brands
    best, score = _best_match(norm, _all_brands_cached())

    # Confidence threshold:
    # - Short strings are tricky; require higher similarity
    # - For typical brand lengths (~6-12), 0.82-0.88 works well
    min_required = 0.84
    if score >= min_required and best:
        # Map brand to category
        for cat, brands, _ in ALL_SETS:
            if best in brands:
                return best, cat, score

    return None, None, score if best else None

# ================= Keyword-only fallback =================
def _keyword_match(text_low: str) -> Optional[str]:
    if any(k in text_low for k in UTILITY_KW):
        return "Utilities"
    if any(k in text_low for k in TRANSPORT_KW):
        return "Transportation"
    if any(k in text_low for k in HEALTH_KW):
        return "Health & Wellness"
    if any(k in text_low for k in GROCERY_KW):
        return "Groceries"
    if any(k in text_low for k in FOOD_KW):
        return "Food"
    return None

# ================= Public API =================
def rule_category(ocr_text: str, store: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (category, reason).
    Uses corrected brand if it's confident; else falls back to keywords in OCR text.
    """
    text_low = (ocr_text or "").lower()
    canon, cat_from_brand, score = correct_store_name(store)

    if cat_from_brand:
        return cat_from_brand, f"brand:{canon}|score:{score:.3f}"

    # Fallback to keywords if no confident brand snap
    cat = _keyword_match(text_low)
    if cat:
        return cat, "keywords"

    return None, None
