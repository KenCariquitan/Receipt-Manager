# app/api.py
from __future__ import annotations

import os, re, asyncio, uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
from joblib import load, dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# --- project locals ---
from .ocr import ocr_image_path, ocr_crop
from .parser import parse_fields, extract_date
from .ph_rules import rule_category, normalize_store_name
from .detect import detect_fields
from .db import (
    init_db, insert_receipt, list_receipts,
    stats_by_category, stats_by_month, stats_summary,
    SessionLocal, Receipt
)


import httpx
from dotenv import load_dotenv
load_dotenv()  # load .env if present

OCR_SPACE_URL = os.getenv("OCR_SPACE_URL", "https://api.ocr.space/parse/image")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OCR_SPACE_ENABLED = os.getenv("OCR_SPACE_ENABLED", "false").lower() == "true"

# ----------------- constants / paths -----------------
DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = Path(__file__).resolve().parents[1] / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

VPATH = MODELS / "vectorizer.joblib"
CPATH = MODELS / "classifier.joblib"
FPATH = DATA / "feedback.csv"

# Keep the public list (used for rules UI etc.)
# Note: The trained classifier may have a different set; we'll read its classes_ at runtime.
CATS_PUBLIC = ["Utilities", "Food", "Groceries", "Transportation", "Health & Wellness", "Others"]

app = FastAPI(title="Receipt Thesis Backend", version="1.3.0")
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
    {"ok": bool, "text": str, "raw": dict|None, "error": str|None}
    """
    if not OCR_SPACE_ENABLED or not OCR_SPACE_API_KEY:
        return {"ok": False, "text": "", "raw": None, "error": "disabled_or_no_key"}

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
            j = r.json()
    except Exception as e:
        return {"ok": False, "text": "", "raw": None, "error": str(e)}

    text = ""
    if isinstance(j, dict) and j.get("ParsedResults"):
        text = "\n".join(pr.get("ParsedText", "") for pr in j["ParsedResults"]).strip()

    ok = (not j.get("IsErroredOnProcessing")) and bool(text)
    return {"ok": ok, "text": text or "", "raw": j if ok else j, "error": None if ok else "api_error"}


# ================== reconcile fields ===================
def resolve_fields(tess_text: str, tess_conf: Optional[float], ocrs: dict) -> tuple[Optional[str], Optional[float], Optional[str], str]:
    """
    Compare Tesseract vs OCR.space parse and produce final (store, total, date, source_tag)
    Priority policy:
      - If OCR.space ok and disagrees, prefer OCR.space.
      - If both agree (roughly), mark 'consensus'.
      - If OCR.space fails/disabled, use Tesseract.
    """
    # Parse tess
    t_store, t_total, t_date = parse_fields(tess_text)

    s_store = s_total = s_date = None
    if ocrs.get("ok") and ocrs.get("text"):
        s_store, s_total, s_date = parse_fields(ocrs["text"])

    def eq_store(a, b):
        if not a or not b: return False
        return a.strip().upper() == b.strip().upper()

    def close_amt(a, b):
        if a is None or b is None: return False
        try:
            return abs(float(a) - float(b)) <= 0.01
        except Exception:
            return False

    agree_store = eq_store(t_store, s_store)
    agree_total = close_amt(t_total, s_total)
    agree_date  = (t_date == s_date) and (t_date is not None)

    if ocrs.get("ok"):
        # Prefer OCR.space when there's any disagreement or when Tesseract is weak
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
    category: str | None = None      # must be in your allowed list (not enforced here)


# ================== Routes ===================
@app.get("/health")
def health():
    classes = []
    if clf is not None and hasattr(clf, "classes_"):
        classes = list(map(str, clf.classes_))
    return {"ok": True, "has_model": bool(clf is not None), "model_classes": classes, "ocr_space": OCR_SPACE_ENABLED}


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
async def upload_receipt(file: UploadFile = File(...)):
    # Save image to disk
    fname = file.filename or "unknown.jpg"
    tmp = DATA / "raw_images" / f"{uuid.uuid4().hex}_{fname}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    tmp.write_bytes(content)

    # ---------- Optional YOLO detect (best-effort) ----------
    img = cv2.imdecode(np.frombuffer(content, dtype='uint8'), cv2.IMREAD_COLOR) if 'np' in globals() else cv2.imread(str(tmp))
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
                m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))", raw.replace("PHP", "").replace("Php", ""))
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
    #Save credits by skipping OCR.space if YOLO was very confident
    tess_text = rec["text"]
    tess_conf = rec.get("mean_conf", 0.0)
    store_t, total_t, date_t = parse_fields(tess_text)

    need_ocr_space = False
    # Heuristic: low confidence or missing critical fields
    if (tess_conf is None or tess_conf < 45) or (store_t is None or total_t is None):
        need_ocr_space = True

    ocrs = {"ok": False, "text": ""}
    if OCR_SPACE_ENABLED and need_ocr_space:
        ocrs = await ocr_space_bytes(content, filename=file.filename or "receipt.jpg")
    else:
        # no need to spend credits
        pass
    # ---------- Parallel OCR.space & reconciliation ----------
    loop = asyncio.get_event_loop()
    ocrs = await (ocr_space_bytes(content, filename=fname) if OCR_SPACE_ENABLED else asyncio.sleep(0, result={"ok": False, "text": ""}))
    store_r, total_r, date_r, source_tag = resolve_fields(rec["text"], rec.get("mean_conf"), ocrs)

    # Prefer YOLO crops when they exist (override reconciled fields if very confident)
    # Prefer YOLO crops when they exist (override reconciled fields if very confident)
    store = yolo_store or store_r
    total = yolo_total if yolo_total is not None else total_r
    date_iso = yolo_date or date_r

    store_norm = normalize_store_name(store) if store else None

    # ---------- Category via rules → ML ----------
    category = confidence = source = reason = None
    cat_rule, reason = rule_category(rec["text"], store_norm or store)
    if cat_rule:
        category, confidence, source = cat_rule, 0.99, "rule"
    elif vectorizer is not None and clf is not None:
        X = vectorizer.transform([rec["text"]])
        proba = getattr(clf, "predict_proba")(X)[0]
        pred_idx = int(proba.argmax())
        if hasattr(clf, "classes_"):
            category = str(clf.classes_[pred_idx])
        else:
            category = CATS_PUBLIC[pred_idx]
        confidence = float(proba.max())
        source = "ml"

    # ---------- Save to DB ----------
    insert_receipt(
        _id=tmp.stem,
        store=store,
        store_norm=store_norm,
        date_iso=date_iso,
        total=total,
        category=category,
        category_source=source,
        confidence=confidence,
        ocr_conf=rec.get("mean_conf"),
        text=rec["text"]
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
        "text": rec["text"],
        "ocr_conf": rec.get("mean_conf"),
        "yolo_used": bool(fields),
        "ocr_space_used": OCR_SPACE_ENABLED,
        "ocr_space_ok": ocrs.get("ok", False),
        "ocr_source": source_tag,  # "tesseract" | "ocr_space" | "consensus"
    }


@app.post("/feedback")
async def feedback(text: str = Form(...), true_label: str = Form(...)):
    # Use PUBLIC list for quick validation; model may have fewer/more
    if true_label not in CATS_PUBLIC:
        return {"ok": False, "msg": f"true_label must be one of {CATS_PUBLIC}"}
    row = pd.DataFrame([[text, true_label]], columns=["text", "label"])
    if FPATH.exists():
        row.to_csv(FPATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FPATH, index=False)
    return {"ok": True}


@app.post("/retrain_incremental")
async def retrain_incremental():
    global clf, vectorizer
    if vectorizer is None or clf is None:
        return {"ok": False, "msg": "Train a base model first (train/train.py)."}
    if not FPATH.exists():
        return {"ok": False, "msg": "No feedback yet."}
    fb = pd.read_csv(FPATH).dropna(subset=["text", "label"])
    # Remap to existing model classes if needed
    X = vectorizer.transform(fb.text.fillna(""))
    classes = list(map(str, getattr(clf, "classes_", CATS_PUBLIC)))
    clf.partial_fit(X, fb.label, classes=classes)
    dump(clf, CPATH)
    return {"ok": True, "count": int(len(fb))}


@app.get("/receipts")
def get_receipts(limit: int = 50, offset: int = 0):
    rows = list_receipts(limit=limit, offset=offset)
    return [{
        "id": r.id,
        "store": r.store,
        "store_normalized": r.store_normalized,
        "date": r.date.isoformat() if r.date is not None else None,
        "total": r.total,
        "category": r.category,
        "category_source": r.category_source,
        "confidence": r.confidence,
        "ocr_conf": r.ocr_conf,
        "created_at": r.created_at.isoformat()
    } for r in rows]


@app.get("/stats/summary")
def get_stats_summary():
    return stats_summary()


@app.get("/stats/by_category")
def get_stats_by_category():
    return stats_by_category()


@app.get("/stats/by_month")
def get_stats_by_month(year: int):
    return stats_by_month(year)


@app.patch("/receipts/{rid}")
def update_receipt(rid: str, upd: ReceiptUpdate):
    with SessionLocal() as db:
        q = db.query(Receipt).filter(Receipt.id == rid)
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


# Ensure DB tables exist on import
init_db()

# --------- numpy import late (avoid import order issues) ----------
# (cv2.imdecode path above needs numpy; import after FastAPI app creation to avoid lints)
import numpy as np  # noqa: E402
