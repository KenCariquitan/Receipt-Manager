import re
from typing import Optional, Tuple

FOOD_BRANDS = {"JOLLIBEE","MCDONALD","MCDONALD'S","KFC","CHOWKING","GREENWICH",
    "MANG INASAL","SHAKEY'S","BONCHON","STARBUCKS","GONG CHA","CHATIME"}
GROCERY_BRANDS = {"SM SUPERMARKET","SM HYPERMARKET","PUREGOLD","ROBINSONS SUPERMARKET","WALTERMART","LANDERS","S&R"}
UTILITY_BRANDS = {"MERALCO","PLDT","GLOBE","SMART","CONVERGE","MAYNILAD","MANILA WATER","SKY","DITO"}
TRANSPORT_BRANDS = {"PETRON","SHELL","CALTEX","SEAOIL","EASYTRIP","AUTOSWEEP","GRAB","ANGKAS","NLEX","SLEX"}
HEALTH_BRANDS = {"MERCURY DRUG","WATSONS","SOUTHSTAR","GENERIKA","ROSE PHARMACY","THE GENERICS PHARMACY"}

UTILITY_KW = {"kwh","kilowatt","meter","account no","service period","due date","statement","internet","fiber","dsl","postpaid","prepaid load","load","data pack","billing"}
TRANSPORT_KW = {"diesel","unleaded","gasoline","pump","liter","litre","toll","rfid","easytrip","autosweep","plate","odometer","grab","angkas"}
FOOD_KW = {"meal","combo","burger","fries","chicken","rice","drink","beverage","snack","dine","take out"}
HEALTH_KW = {"pharmacy","rx","tablet","capsule","mg","ml","clinic","dental","optical","laboratory","prescription"}
GROCERY_KW = {"grocery","supermarket"}

ALL_SETS = [
    ("Food", FOOD_BRANDS, FOOD_KW),
    ("Utilities", UTILITY_BRANDS, UTILITY_KW),
    ("Transportation", TRANSPORT_BRANDS, TRANSPORT_KW),
    ("Health & Wellness", HEALTH_BRANDS, HEALTH_KW),
]

SPACES = re.compile(r"\s+")

def normalize_store_name(store: Optional[str]) -> Optional[str]:
    if not store:
        return None
    s = store.upper()
    s = SPACES.sub(" ", s)
    s = re.sub(r"\b(CORP(?:ORATION)?|INC\.?|CO\.?|COMPANY|LTD\.?|CORPORATION)\b","",s)
    s = re.sub(r"\s{2,}"," ",s).strip()
    return s

def _store_match(norm: str) -> Optional[str]:
    for cat, brands, _ in ALL_SETS:
        for b in brands:
            if b in norm:
                return cat
    for b in GROCERY_BRANDS:
        if b in norm:
            return "Food"
    return None

def _keyword_match(text_low: str) -> Optional[str]:
    if any(k in text_low for k in UTILITY_KW):
        return "Utilities"
    if any(k in text_low for k in TRANSPORT_KW):
        return "Transportation"
    if any(k in text_low for k in HEALTH_KW):
        return "Health & Wellness"
    if any(k in text_low for k in FOOD_KW) or any(k in text_low for k in GROCERY_KW):
        return "Food"
    return None

def rule_category(ocr_text: str, store: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    text_low = ocr_text.lower()
    norm = normalize_store_name(store) if store else None
    if norm:
        cat = _store_match(norm)
        if cat:
            return cat, f"brand-match:{norm}"
    cat = _keyword_match(text_low)
    if cat:
        return cat,"keywords"
    return None,None
