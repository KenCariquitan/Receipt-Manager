# app/api.py
from __future__ import annotations

import os, re, uuid, asyncio, math
from pathlib import Path
from typing import Optional
from jose import jwt
from jose.utils import base64url_decode
from jose.backends.cryptography_backend import CryptographyRSAKey
import httpx

import numpy as np
import pandas as pd
import cv2
from math import isnan, isinf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load, dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# --- project locals ---
from .ocr import ocr_image_path, ocr_crop
from .parser import parse_fields, extract_date  # improved total/date parsing
from .ph_rules import rule_category, normalize_store_name, correct_store_name
from .detect import detect_fields
from .db import (
    init_db, insert_receipt, list_receipts,
    stats_by_category, stats_by_month, stats_summary,
    SessionLocal, Receipt
)

import httpx
from dotenv import load_dotenv
from jose import jwt

load_dotenv()  # load .env if present

# ================== helpers for floats ==================
def _nan_none(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def _clean_num(x):
    try:
        return None if x is None or isnan(x) or isinf(x) else float(x)
    except Exception:
        return None

# ================== OCR.space config ==================
OCR_SPACE_URL = os.getenv("OCR_SPACE_URL", "https://api.ocr.space/parse/image")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OCR_SPACE_ENABLED = os.getenv("OCR_SPACE_ENABLED", "false").lower() == "true"

# ================== Supabase Auth (JWT) ==================

SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL")
SUPABASE_PROJECT_REF= os.getenv("SUPABASE_PROJECT_REF")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_JWKS_CACHE: Optional[dict] = None  

async def _get_jwks():
    global _JWKS_CACHE
    if _JWKS_CACHE is None:
        if not SUPABASE_JWKS_URL:
            raise RuntimeError("SUPABASE_PROJECT_REF not set")
        headers = {"apikey": SUPABASE_ANON_KEY} if SUPABASE_ANON_KEY else {}
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(SUPABASE_JWKS_URL)
            esp = await c.get(SUPABASE_JWKS_URL, headers=headers)
            resp.raise_for_status()
            _JWKS_CACHE = resp.json()
    return _JWKS_CACHE

def _get_kid(token: str) -> Optional[str]:
    try:
        header = jwt.get_unverified_header(token)
        return header.get("kid")
    except Exception:
        return None
async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    payload = jwt.get_unverified_claims(token)  # WARNING: no signature check
    return payload

    # # Get JWKS from Supabase
    # async with httpx.AsyncClient(timeout=10) as client:
    #     resp = await client.get(SUPABASE_JWKS_URL)
    #     resp.raise_for_status()
    #     jwks = resp.json()

    # # Match kid
    # header = jwt.get_unverified_header(token)
    # kid = header.get("kid")
    # jwk = next((k for k in jwks["keys"] if k["kid"] == kid), None)
    # if not jwk:
    #     raise HTTPException(status_code=401, detail="Invalid token (kid)")

    # # Convert JWK → RSA key
    # public_key = CryptographyRSAKey(jwk)

    # try:
    #     payload = jwt.decode(
    #         token,
    #         public_key,
    #         algorithms=["RS256"],
    #         options={"verify_aud": False}
    #     )
    # except Exception as e:
    #     raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    # return payload  # has "sub" (user_id), "email", etc.

# ----------------- constants / paths -----------------
DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = Path(__file__).resolve().parents[1] / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

VPATH = MODELS / "vectorizer.joblib"
CPATH = MODELS / "classifier.joblib"
FPATH = DATA / "feedback.csv"

# Public list for validation/UI; classifier classes_ may differ at runtime
CATS_PUBLIC = ["Utilities", "Food", "Groceries", "Transportation", "Health & Wellness", "Others"]

app = FastAPI(title="Receipt Thesis Backend", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------- lazy-load model (if present) -------------
vectorizer: Optional[TfidfVectorizer] = load(VPATH) if VPATH.exists() else None
clf: Optional[SGDClassifier] = load(CPATH) if CPATH.exists() else None

# ================== OCR.space helper ===================
async def ocr_space_bytes(img_bytes: bytes, filename: str = "receipt.jpg", lang: str = "eng") -> dict:
    """
    Call OCR.space with image bytes. Returns:
      {"ok": bool, "text": str, "raw": dict|None, "error": str|None, "http": int|None}
    """
    if not OCR_SPACE_ENABLED or not OCR_SPACE_API_KEY:
        return {"ok": False, "text": "", "raw": None, "error": "disabled_or_no_key", "http": None}

    data = {
        "language": lang,
        "isOverlayRequired": False,
        "OCREngine": 2,
        "scale": True,
        "isTable": False
    }
    headers = {"apikey": OCR_SPACE_API_KEY}
    files = {"file": (filename, img_bytes, "application/octet-stream")}

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(OCR_SPACE_URL, data=data, headers=headers, files=files)
            http_code = r.status_code
            j = r.json()
    except Exception as e:
        return {"ok": False, "text": "", "raw": None, "error": f"network:{e}", "http": None}

    text = ""
    if isinstance(j, dict) and j.get("ParsedResults"):
        text = "\n".join(pr.get("ParsedText", "") for pr in j["ParsedResults"]).strip()

    err = None
    if isinstance(j, dict) and j.get("IsErroredOnProcessing"):
        err = f"api:{j.get('ErrorMessage') or j.get('ErrorDetails') or 'unknown'}"

    ok = bool(text) and not j.get("IsErroredOnProcessing", False)
    return {"ok": ok, "text": text or "", "raw": j, "error": err, "http": http_code}

# ================== reconcile fields ===================
def resolve_fields(tess_text: str, tess_conf: Optional[float], ocrs: dict) -> tuple[Optional[str], Optional[float], Optional[str], str]:
    """
    Compare Tesseract vs OCR.space parsing (both pass through the same parser)
    and produce final (store, total, date, source_tag).
    Policy:
      - If OCR.space ok and disagrees, prefer OCR.space.
      - If both agree, mark 'consensus'.
      - If OCR.space fails/disabled, use Tesseract.
    """
    # Parse Tesseract text
    t_store, t_total, t_date = parse_fields(tess_text)

    s_store = s_total = s_date = None
    if ocrs.get("ok") and ocrs.get("text"):
        s_store, s_total, s_date = parse_fields(ocrs["text"])

    def eq_store(a, b):
        if not a or not b:
            return False
        return a.strip().upper() == b.strip().upper()

    def close_amt(a, b):
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) <= 0.01
        except Exception:
            return False

    agree_store = eq_store(t_store, s_store)
    agree_total = close_amt(t_total, s_total)
    agree_date = (t_date == s_date) and (t_date is not None)

    if ocrs.get("ok"):
        if not (agree_store and agree_total and agree_date):
            source = "ocr_space"
            store = s_store or t_store
            total = s_total if s_total is not None else t_total
            date_iso = s_date or t_date
        else:
            source = "consensus"
            store, total, date_iso = t_store, t_total, t_date
    else:
        source = "tesseract"
        store, total, date_iso = t_store, t_total, t_date

    return store, total, date_iso, source

# ================== API Schemas ===================
class TextIn(BaseModel):
    text: str

class ReceiptUpdate(BaseModel):
    store: str | None = None
    date: str | None = None          # ISO YYYY-MM-DD
    total: float | None = None
    category: str | None = None      # validated in UI; not enforced here

# ================== Routes ===================
@app.get("/health")
def health():
    classes = []
    if clf is not None and hasattr(clf, "classes_"):
        classes = list(map(str, clf.classes_))
    return {
        "ok": True,
        "has_model": bool(clf is not None),
        "model_classes": classes,
        "ocr_space": OCR_SPACE_ENABLED,
        "auth": bool(SUPABASE_PROJECT_REF),
    }

@app.post("/classify_text")
def classify_text(inp: TextIn):
    # 1) Rule-based first
    cat_rule, reason = rule_category(inp.text, None)
    if cat_rule:
        return {"pred": cat_rule, "proba": {}, "source": "rule", "reason": reason}

    # 2) ML next
    if vectorizer is None or clf is None:
        return {"error": "Model not trained yet."}

    X = vectorizer.transform([inp.text])
    proba = getattr(clf, "predict_proba")(X)[0]
    pred_idx = int(proba.argmax())

    # Use model's own classes_ to avoid mismatch with CATS_PUBLIC
    if hasattr(clf, "classes_"):
        label = str(clf.classes_[pred_idx])
        cls_list = list(map(str, clf.classes_))
    else:
        label = CATS_PUBLIC[pred_idx]
        cls_list = CATS_PUBLIC

    return {"pred": label, "proba": {c: float(p) for c, p in zip(cls_list, proba)}, "source": "ml", "reason": None}

@app.post("/upload_receipt")
async def upload_receipt(file: UploadFile = File(...), user=Depends(get_current_user)):
    # Auth
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id in token")

    # Save image to disk
    fname = file.filename or "unknown.jpg"
    tmp = DATA / "raw_images" / f"{uuid.uuid4().hex}_{fname}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    tmp.write_bytes(content)

    # ---------- Optional YOLO detect (best-effort) ----------
    try:
        img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(str(tmp))

    fields = []
    yolo_store = yolo_total = yolo_date = None
    try:
        fields = detect_fields(img)  # [] if model missing or no detections
        if fields:
            # Merchant
            best = max([f for f in fields if f["name"] == "Merchant"], key=lambda x: x["conf"], default=None)
            if best:
                yolo_store = ocr_crop(img, best["box"], psm=7)

            # Total
            best = max([f for f in fields if f["name"] == "Total"], key=lambda x: x["conf"], default=None)
            if best:
                raw = ocr_crop(img, best["box"], psm=7, allowlist="0123456789.,₱PHPPhp ")
                m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))",
                              raw.replace("PHP", "").replace("Php", ""))
                if m:
                    yolo_total = float(m.group(1).replace(",", ""))

            # Date
            best = max([f for f in fields if f["name"] == "Date"], key=lambda x: x["conf"], default=None)
            if best:
                date_txt = ocr_crop(img, best["box"], psm=6)
                yolo_date = extract_date(date_txt)
    except Exception:
        fields = []  # YOLO is optional; swallow errors

    # ---------- Full-page OCR (Tesseract) ----------
    rec = ocr_image_path(str(tmp))  # {path, text, mean_conf, w, h}
    tess_text = rec["text"]
    tess_conf = rec.get("mean_conf", 0.0)

    # ---------- Decide if we should call OCR.space (save credits) ----------
    store_t, total_t, date_t = parse_fields(tess_text)
    need_ocr_space = (tess_conf is None or tess_conf < 45) or (store_t is None or total_t is None)

    if OCR_SPACE_ENABLED and need_ocr_space:
        ocrs = await ocr_space_bytes(content, filename=fname)
    else:
        ocrs = {"ok": False, "text": "", "error": None, "http": None}

    # ---------- Reconcile Tesseract vs OCR.space ----------
    store_r, total_r, date_r, source_tag = resolve_fields(tess_text, tess_conf, ocrs)

    # ---------- Prefer YOLO crops when present ----------
    store = yolo_store or store_r
    total = yolo_total if yolo_total is not None else total_r
    date_iso = yolo_date or date_r

    store_norm = normalize_store_name(store) if store else None

    # Fuzzy brand correction (e.g., "¢-ELEWEOD" -> "7-ELEVEN")
    canon, canon_cat, canon_score = correct_store_name(store_norm or store)
    if canon and canon_score and canon_score >= 0.86:
        store = canon
        store_norm = canon

    # ---------- Category via rules → ML ----------
    category = confidence = source = reason = None
    cat_rule_val, reason = rule_category(tess_text, store_norm or store)
    if cat_rule_val:
        category, confidence, source = cat_rule_val, 0.99, "rule"
    elif vectorizer is not None and clf is not None:
        X = vectorizer.transform([tess_text])
        proba = getattr(clf, "predict_proba")(X)[0]
        pred_idx = int(proba.argmax())
        if hasattr(clf, "classes_"):
            category = str(clf.classes_[pred_idx])
        else:
            category = CATS_PUBLIC[pred_idx]
        confidence = float(proba.max())
        source = "ml"

    # ---------- Save to DB (per user) ----------
    insert_receipt(
        _id=tmp.stem,
        user_id=user_id,                 # <<< IMPORTANT
        store=store,
        store_norm=store_norm,
        date_iso=date_iso,
        total=_clean_num(total),
        category=category,
        category_source=source,
        confidence=_clean_num(confidence),
        ocr_conf=_clean_num(rec.get("mean_conf")),
        text=tess_text
    )

    return {
        "id": tmp.stem,
        "store": store,
        "store_normalized": store_norm,
        "date": date_iso,
        "total": total,
        "category": category,
        "confidence": confidence,
        "category_source": source,
        "reason": reason,
        "text": tess_text,
        "ocr_conf": rec.get("mean_conf"),
        "yolo_used": bool(fields),
        "ocr_space_used": OCR_SPACE_ENABLED and need_ocr_space,
        "ocr_space_ok": ocrs.get("ok", False),
        "ocr_space_http": ocrs.get("http"),
        "ocr_space_err": ocrs.get("error"),
        "ocr_source": source_tag,  # "tesseract" | "ocr_space" | "consensus"
    }

@app.post("/feedback")
async def feedback(text: str = Form(...), true_label: str = Form(...), user=Depends(get_current_user)):
    # You could store user_id along with the feedback if you want per-user auditing
    if true_label not in CATS_PUBLIC:
        return {"ok": False, "msg": f"true_label must be one of {CATS_PUBLIC}"}
    row = pd.DataFrame([[text, true_label]], columns=["text", "label"])
    if FPATH.exists():
        row.to_csv(FPATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FPATH, index=False)
    return {"ok": True}

@app.post("/retrain_incremental")
async def retrain_incremental(user=Depends(get_current_user)):
    # (kept auth just in case; you can restrict by role/claim)
    global clf, vectorizer
    if vectorizer is None or clf is None:
        return {"ok": False, "msg": "Train a base model first (train/train.py)."}
    if not FPATH.exists():
        return {"ok": False, "msg": "No feedback yet."}
    fb = pd.read_csv(FPATH).dropna(subset=["text", "label"])
    X = vectorizer.transform(fb.text.fillna(""))
    classes = list(map(str, getattr(clf, "classes_", CATS_PUBLIC)))
    clf.partial_fit(X, fb.label, classes=classes)
    dump(clf, CPATH)
    return {"ok": True, "count": int(len(fb))}

@app.get("/receipts")
def get_receipts(limit: int = 50, offset: int = 0, user=Depends(get_current_user)):
    rows = list_receipts(user_id=user.get("sub"), limit=limit, offset=offset)
    return [{
        "id": r.id,
        "store": r.store,
        "store_normalized": r.store_normalized,
        "date": r.date.isoformat() if r.date is not None else None,
        "total": _nan_none(r.total),
        "category": r.category,
        "category_source": r.category_source,
        "confidence": _nan_none(r.confidence),
        "ocr_conf": _nan_none(r.ocr_conf),
        "created_at": r.created_at.isoformat()
    } for r in rows]

@app.get("/stats/summary")
def get_stats_summary(user=Depends(get_current_user)):
    return stats_summary(user_id=user.get("sub"))

@app.get("/stats/by_category")
def get_stats_by_category(user=Depends(get_current_user)):
    return stats_by_category(user_id=user.get("sub"))

@app.get("/stats/by_month")
def get_stats_by_month(year: int, user=Depends(get_current_user)):
    return stats_by_month(year, user_id=user.get("sub"))

@app.patch("/receipts/{rid}")
def update_receipt(rid: str, upd: ReceiptUpdate, user=Depends(get_current_user)):
    with SessionLocal() as db:
        q = db.query(Receipt).filter(Receipt.id == rid, Receipt.user_id == user.get("sub"))
        r = q.first()
        if not r:
            raise HTTPException(status_code=404, detail="not found")

        data = upd.model_dump(exclude_unset=True)
        if "date" in data and data["date"] is not None:
            from datetime import date as _d
            try:
                data["date"] = _d.fromisoformat(data["date"])
            except Exception:
                data["date"] = None

        for k, v in data.items():
            setattr(r, k, v)
        db.commit()
    return {"ok": True}
@app.get("/debug/token")
async def debug_token(request: Request):
    auth = request.headers.get("authorization")
    return {"auth_header": auth}
# Ensure DB tables exist on import
init_db()
