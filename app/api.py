from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from joblib import load, dump
import pandas as pd
import uuid
import cv2  

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .ocr import ocr_image_path, ocr_crop
from .parser import parse_fields, extract_date
from .ph_rules import rule_category, normalize_store_name
from .detect import detect_fields  
from .db import init_db, insert_receipt, list_receipts, stats_by_category, stats_by_month, stats_summary
from fastapi import HTTPException
from pydantic import BaseModel
from .db import SessionLocal, Receipt


DATA = Path(__file__).resolve().parents[1]/"data"
MODELS = Path(__file__).resolve().parents[1]/"models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

VPATH = MODELS/"vectorizer.joblib"
CPATH = MODELS/"classifier.joblib"
FPATH = DATA/"feedback.csv"

# If you later add "Groceries", keep the same order across files
CATS = ["Utilities","Food","Transportation","Health & Wellness","Others"]

app = FastAPI(title="Receipt Thesis Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load models (may not exist on first run)
vectorizer: TfidfVectorizer | None = load(VPATH) if VPATH.exists() else None
clf: SGDClassifier | None = load(CPATH) if CPATH.exists() else None

class TextIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True, "has_model": bool(clf is not None)}

@app.post("/classify_text")
def classify_text(inp: TextIn):
    # 1) Rule-based first
    cat, reason = rule_category(inp.text, None)
    if cat:
        return {"pred": cat, "proba": {}, "source": "rule", "reason": reason}
    # 2) ML next
    if vectorizer is None or clf is None:
        return {"error": "Model not trained yet."}
    X = vectorizer.transform([inp.text])
    proba = getattr(clf, "predict_proba")(X)[0]
    pred_idx = int(proba.argmax())
    return {"pred": CATS[pred_idx], "proba": {c: float(p) for c,p in zip(CATS, proba)}, "source": "ml"}

@app.post("/upload_receipt")
async def upload_receipt(file: UploadFile = File(...)):
    # Save image
    tmp = DATA/"raw_images"/f"{uuid.uuid4().hex}_{file.filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())

    # ---- YOLOv8: detect regions first ----
    img = cv2.imread(str(tmp))
    fields = []  
    yolo_store = yolo_total = yolo_date = None

    try:
        fields = detect_fields(img)  # [] if model missing or no detections
        if fields:
            # store_header (PSM 7 single line)
            best = max([f for f in fields if f["name"]=="Merchant"], key=lambda x: x["conf"], default=None)
            if best:
                yolo_store = ocr_crop(img, best["box"], psm=7)

            # total_value (force digits & currency)
            best = max([f for f in fields if f["name"]=="Total"], key=lambda x: x["conf"], default=None)
            if best:
                raw = ocr_crop(img, best["box"], psm=7, allowlist="0123456789.,₱PHPPhp ")
                import re
                m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2})|[0-9]+(?:\\.[0-9]{2}))",
                              raw.replace("PHP","").replace("Php",""))
                if m:
                    yolo_total = float(m.group(1).replace(",",""))

            # date_box (PSM 6)
            best = max([f for f in fields if f["name"]=="Date"], key=lambda x: x["conf"], default=None)
            if best:
                date_txt = ocr_crop(img, best["box"], psm=6)
                yolo_date = extract_date(date_txt)

    except Exception:
        # YOLO is optional; any error just falls back below
        pass
        
    # ---- Full-page OCR as fallback / fill missing ----
    rec = ocr_image_path(str(tmp))  # {text, mean_conf, ...}
    # your existing parser (text-only)
    store_fallback, total_fallback, date_fallback = parse_fields(rec["text"])

    store = yolo_store or store_fallback
    total = yolo_total if yolo_total is not None else total_fallback
    date_iso = yolo_date or date_fallback

    store_norm = normalize_store_name(store) if store else None

    # ---- Category via rules or ML ----
    category = confidence = source = reason = None
    cat_rule, reason = rule_category(rec["text"], store_norm or store)
    if cat_rule:
        category, confidence, source = cat_rule, 0.99, "rule"
    elif vectorizer is not None and clf is not None:
        X = vectorizer.transform([rec["text"]])
        proba = getattr(clf, "predict_proba")(X)[0]
        category = CATS[int(proba.argmax())]
        confidence = float(proba.max())
        source = "ml"
        
    insert_receipt(
        _id=tmp.stem,
        store=store,
        store_norm=store_norm,
        date_iso=date_iso,
        total=total,
        category=category,
        category_source=source,
        confidence=confidence,
        ocr_conf=rec["mean_conf"],
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
        "ocr_conf": rec["mean_conf"],
        "yolo_used": bool(fields),  # helpful for debugging
    }

@app.post("/feedback")
async def feedback(text: str = Form(...), true_label: str = Form(...)):
    if true_label not in CATS:
        return {"ok": False, "msg": f"true_label must be one of {CATS}"}
    row = pd.DataFrame([[text, true_label]], columns=["text","label"])
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
    fb = pd.read_csv(FPATH)
    X = vectorizer.transform(fb.text.fillna(""))
    classes = CATS
    clf.partial_fit(X, fb.label, classes=classes)
    dump(clf, CPATH)
    return {"ok": True, "count": len(fb)}


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

class ReceiptUpdate(BaseModel):
    store: str | None = None
    date: str | None = None
    total: float | None = None
    category: str | None = None

@app.patch("/receipts/{rid}")
def update_receipt(rid: str, upd: ReceiptUpdate):
    with SessionLocal() as db:
        receipt_query = db.query(Receipt).filter(Receipt.id == rid)
        r = receipt_query.first()
        if not r:
            raise HTTPException(status_code=404, detail="not found")

        update_data = upd.model_dump(exclude_unset=True)
        
        if 'date' in update_data and update_data['date'] is not None:
            from datetime import date as _d
            try:
                update_data['date'] = _d.fromisoformat(update_data['date'])
            except (ValueError, TypeError):
                update_data['date'] = None
        
        if update_data:
            for key, value in update_data.items():
                setattr(r, key, value)
            db.commit()
            
    return {"ok": True}


# Ensure DB tables exist
init_db()