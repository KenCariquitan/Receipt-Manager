from __future__ import annotations
import re
from typing import Optional, Tuple, Iterable
import Levenshtein as lev  # pip install python-Levenshtein

# ================= Canonical brand sets (UPPERCASE) =================
FOOD_BRANDS = {
    "JOLLIBEE","MCDONALD","MCDONALD'S","KFC","CHOWKING","GREENWICH",
    "MANG INASAL","SHAKEY'S","BONCHON","STARBUCKS","GONG CHA","CHATIME",
    "7-ELEVEN", "MINISTOP", "FAMILYMART"
}
GROCERY_BRANDS = {
    "SM SUPERMARKET","SM HYPERMARKET","PUREGOLD","ROBINSONS SUPERMARKET",
    "WALTERMART","LANDERS","S&R"
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

def normalize_store_name(store: Optional[str]) -> Optional[str]:
    if not store:
        return None
    s = _sanitize_ocr(store)
    # Remove legal suffixes
    s = re.sub(r"\b(CORP(?:ORATION)?|INC\.?|CO\.?|COMPANY|LTD\.?|CORPORATION)\b", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# ================= Fuzzy snapping to canonical brand =================
def _distance(a: str, b: str) -> int:
    return lev.distance(a, b)

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
        d = _distance(norm, c)
        denom = max(len(norm), len(c)) or 1
        score = 1.0 - (d / denom)
        if score > best_score:
            best, best_score = c, score
    return best, best_score

def _all_brands() -> list[str]:
    brands: list[str] = []
    for _, bset, _ in ALL_SETS:
        brands.extend(list(bset))
    return brands

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

    # 1) Exact/contains quick pass
    for cat, brands, _ in ALL_SETS:
        for b in brands:
            if b in norm or norm in b:
                return b, cat, 1.0

    # 2) Fuzzy against all brands
    best, score = _best_match(norm, _all_brands())

    # Confidence threshold:
    # - Short strings are tricky; require higher similarity
    # - For typical brand lengths (~6-12), 0.82–0.88 works well
    min_required = 0.86
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
