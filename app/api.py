from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from joblib import load, dump
import pandas as pd
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .ocr import ocr_image_path
from .parser import parse_fields
from .ph_rules import rule_category, normalize_store_name

DATA = Path(__file__).resolve().parents[1]/"data"
MODELS = Path(__file__).resolve().parents[1]/"models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

VPATH = MODELS/"vectorizer.joblib"
CPATH = MODELS/"classifier.joblib"
FPATH = DATA/"feedback.csv"

CATS = ["Utilities","Food","Transportation","Health & Wellness","Others"]

app = FastAPI(title="Receipt Thesis Backend", version="1.0.0")
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
    # Rule-based first
    cat, reason = rule_category(inp.text, None)
    if cat:
        return {"pred": cat, "proba": {}, "source": "rule", "reason": reason}
    # ML next
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

    # OCR + parse
    rec = ocr_image_path(str(tmp))
    store, total, date_iso = parse_fields(rec["text"])
    store_norm = normalize_store_name(store) if store else None

    # Category via rules or ML
    category, confidence, source, reason = None, None, None, None
    cat_rule, reason = rule_category(rec["text"], store_norm or store)
    if cat_rule:
        category, confidence, source = cat_rule, 0.99, "rule"
    elif vectorizer is not None and clf is not None:
        X = vectorizer.transform([rec["text"]])
        proba = getattr(clf, "predict_proba")(X)[0]
        category = CATS[int(proba.argmax())]
        confidence = float(proba.max())
        source = "ml"

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
        return {"ok": False, "msg": "Train a base model first (train/train.py)."},
    if not FPATH.exists():
        return {"ok": False, "msg": "No feedback yet."}
    fb = pd.read_csv(FPATH)
    X = vectorizer.transform(fb.text.fillna(""))
    classes = CATS
    clf.partial_fit(X, fb.label, classes=classes)
    dump(clf, CPATH)
    return {"ok": True, "count": len(fb)}
