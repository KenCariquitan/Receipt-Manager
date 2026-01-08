# app/api.py
from __future__ import annotations

import os, re, uuid, asyncio, math, time, logging, json
from pathlib import Path
from typing import Optional
from datetime import date as _date, datetime as _datetime
from jose import jwt
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
from .ocr import ocr_crop, ocr_amount_from_crop
from .ocr_google import GCV_ENABLED
from .ocr_paddle_vl import paddle_vl_text
from .parser import parse_fields, extract_date, parse_fields_from_ocr  # improved total/date parsing
from .ph_rules import rule_category, normalize_store_name, correct_store_name
from .detect import detect_fields
from .db import (
    init_db, insert_receipt, list_receipts,
    stats_by_category, stats_by_month, stats_summary,
    top_merchants_current_month, weekday_spend,
    rolling_30_day_spend, low_confidence_receipts,
    SessionLocal, Receipt, ReceiptCorrection, CustomLabel,
    create_custom_label, list_custom_labels, get_custom_label,
    update_custom_label, delete_custom_label, increment_label_usage,
)
from .ocr_strategies import (
    OCRContext,
    TesseractStrategy,
    OCRSpaceStrategy,
    GoogleVisionStrategy,
    PaddleVLStrategy,
)

from dotenv import load_dotenv

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
TESSERACT_ENABLED = os.getenv("TESSERACT_ENABLED", "true").lower() == "true"
PADDLE_VL_ENABLED = os.getenv("PADDLE_VL_ENABLED", "false").lower() == "true"

_ocr_strategies = {}
if TESSERACT_ENABLED:
    _ocr_strategies["tesseract"] = TesseractStrategy()
_ocr_strategies["ocr_space"] = OCRSpaceStrategy(enabled=OCR_SPACE_ENABLED)
_ocr_strategies["google_vision"] = GoogleVisionStrategy(enabled=GCV_ENABLED)
_ocr_strategies["paddle_vl"] = PaddleVLStrategy(enabled=PADDLE_VL_ENABLED)

JOBS: dict[str, dict] = {}
JOBS_LOCK = asyncio.Lock()

OCR_STRATEGY_CONTEXT = OCRContext(_ocr_strategies)

MAX_JOBS = 200


async def _prune_jobs_locked():
    if len(JOBS) <= MAX_JOBS:
        return
    removable = [
        job for job in JOBS.values()
        if job.get("status") in {"completed", "failed"}
    ]
    removable.sort(key=lambda j: j.get("updated_at", j.get("created_at", 0)))
    for job in removable:
        if len(JOBS) <= MAX_JOBS:
            break
        JOBS.pop(job["id"], None)


async def _register_job(user_id: str, filename: str) -> dict:
    now = time.time()
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "user_id": user_id,
        "filename": filename,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    async with JOBS_LOCK:
        JOBS[job_id] = job
        await _prune_jobs_locked()
    return job


async def _update_job(job_id: str, **changes) -> Optional[dict]:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        job.update(changes)
        job["updated_at"] = time.time()
        return job.copy()


async def _get_job(job_id: str) -> Optional[dict]:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        return job.copy() if job else None


def _public_job(job: dict) -> dict:
    safe = job.copy()
    safe.pop("user_id", None)
    return safe

def _crop_to_bytes(img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> bytes | None:
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return encoded.tobytes()

# ================== Supabase Auth (JWT) ==================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL")
SUPABASE_PROJECT_REF= os.getenv("SUPABASE_PROJECT_REF")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_JWKS_CACHE: Optional[dict] = None  
_USER_CACHE: dict[str, tuple[dict, float]] = {}
_CACHE_GRACE = 10  # seconds to subtract from token expiry when caching
_CACHE_DEFAULT_TTL = 300  # fallback cache TTL (5 minutes)

async def _get_jwks():
    global _JWKS_CACHE
    if not SUPABASE_JWKS_URL:
        return None
    if _JWKS_CACHE is None:
        headers = {"apikey": SUPABASE_ANON_KEY} if SUPABASE_ANON_KEY else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(SUPABASE_JWKS_URL, headers=headers)
            resp.raise_for_status()
            _JWKS_CACHE = resp.json()
    return _JWKS_CACHE

def _get_kid(token: str) -> Optional[str]:
    try:
        header = jwt.get_unverified_header(token)
        return header.get("kid")
    except Exception:
        return None
async def _fetch_supabase_user(token: str) -> dict:
    """
    Fallback auth validation via Supabase REST when JWKS is unavailable.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase URL/anon key not configured")

    user_url = SUPABASE_URL.rstrip("/") + "/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": SUPABASE_ANON_KEY,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(user_url, headers=headers)

    if resp.status_code != 200:
        detail = "Supabase token invalid"
        try:
            data = resp.json()
            detail = data.get("message") or data.get("error_description") or detail
        except Exception:
            pass
        raise HTTPException(status_code=401, detail=detail)

    user = resp.json()
    claims = {}
    try:
        claims = jwt.get_unverified_claims(token)
    except Exception:
        claims = {}

    payload = {
        "sub": user.get("id") or claims.get("sub"),
        "email": user.get("email") or claims.get("email"),
        "app_metadata": user.get("app_metadata") or claims.get("app_metadata", {}),
        "user_metadata": user.get("user_metadata") or claims.get("user_metadata", {}),
        "supabase_user": user,
        **{k: v for k, v in claims.items() if k not in {"sub", "email", "app_metadata", "user_metadata"}},
    }
    return payload

def _cache_lookup(token: str) -> Optional[dict]:
    entry = _USER_CACHE.get(token)
    if not entry:
        return None
    payload, expires_at = entry
    if expires_at is None or expires_at > time.time():
        return payload
    _USER_CACHE.pop(token, None)
    return None

def _cache_store(token: str, payload: dict):
    exp_claim = payload.get("exp")
    ttl = _CACHE_DEFAULT_TTL
    now = time.time()
    if isinstance(exp_claim, (int, float)):
        ttl = max(0, exp_claim - now - _CACHE_GRACE)
    expires_at = now + ttl if ttl > 0 else now + _CACHE_DEFAULT_TTL
    _USER_CACHE[token] = (payload, expires_at)

async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()

    cached = _cache_lookup(token)
    if cached is not None:
        return cached

    payload: Optional[dict] = None

    try:
        jwks = await _get_jwks()
    except Exception:
        jwks = None

    if jwks and jwks.get("keys"):
        kid = _get_kid(token)
        if kid:
            jwk = next((k for k in jwks["keys"] if k.get("kid") == kid), None)
            if jwk is not None:
                public_key = CryptographyRSAKey(jwk)
                try:
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=["RS256"],
                        options={"verify_aud": False},
                    )
                except Exception:
                    payload = None

    if payload is None:
        payload = await _fetch_supabase_user(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    _cache_store(token, payload)

    return payload  # has "sub", "email", etc.

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

def _serialize_value(value):
    if isinstance(value, (_date, _datetime)):
        return value.isoformat()
    if value is None:
        return None
    return str(value)

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
def resolve_fields(
    tess_rec: dict,
    tess_conf: Optional[float],
    ocrs: dict,
    vision: dict,
    paddle: dict | None = None,
) -> tuple[Optional[str], Optional[float], Optional[str], str]:
    """
    Compare outputs from Tesseract, OCR.space, and (optionally) Google Vision.
    Return the best (store, total, date, source_tag).
    """

    def close_amt(a, b):
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) <= 0.01
        except Exception:
            return False

    def eq_store(a, b):
        if not a or not b:
            return False
        return a.strip().upper() == b.strip().upper()

    candidates = []

    tess_text = tess_rec.get("text", "") if isinstance(tess_rec, dict) else str(tess_rec or "")
    t_store, t_total, t_date = parse_fields_from_ocr(tess_rec if isinstance(tess_rec, dict) else {"text": tess_text})
    fb_store, fb_total, fb_date = parse_fields(tess_text)
    t_store = t_store or fb_store
    t_total = t_total if t_total is not None else fb_total
    t_date = t_date or fb_date

    candidates.append({
        "source": "tesseract",
        "store": t_store,
        "total": t_total,
        "date": t_date,
        "confidence": tess_conf or 0.0,
        "priority": 0,
    })

    if ocrs.get("ok") and ocrs.get("text"):
        s_store, s_total, s_date = parse_fields(ocrs["text"])
        candidates.append({
            "source": "ocr_space",
            "store": s_store,
            "total": s_total,
            "date": s_date,
            "confidence": 70.0,  # heuristic confidence
            "priority": 0.5,
        })

    if vision.get("ok") and vision.get("text"):
        v_store, v_total, v_date = parse_fields(vision["text"])
        v_conf = None
        try:
            v_conf = vision.get("info", {}).get("confidence")
        except Exception:
            v_conf = None
        candidates.append({
            "source": "vision",
            "store": v_store,
            "total": v_total,
            "date": v_date,
            "confidence": (v_conf or 0.0) * 100 if isinstance(v_conf, float) else 50.0,
            "priority": 2,
        })

    if paddle:
        candidates.append({
            "source": "paddle_vl",
            "store": paddle.get("store"),
            "total": paddle.get("total"),
            "date": paddle.get("date"),
            "confidence": float(paddle.get("confidence", 85.0)),
            "priority": -0.2,
        })

    # safety: ensure at least tesseract candidate exists
    if not candidates:
        return None, None, None, "unknown"

    def score(cand):
        score = 0.0
        if cand["store"]:
            score += 2.0
        if cand["total"] is not None:
            score += 3.5
        if cand["date"]:
            score += 1.2
        score += min(cand.get("confidence") or 0.0, 100.0) / 40.0
        score += max(0.0, 3.0 - cand["priority"])
        if cand["source"] == "ocr_space":
            score += 0.25
        if cand["source"] == "paddle_vl":
            score += 0.5
        return score

    best = max(candidates, key=score)

    # Detect consensus (any other candidate matching best)
    consensus = False
    for cand in candidates:
        if cand is best:
            continue
        same_store = eq_store(cand["store"], best["store"])
        same_total = close_amt(cand["total"], best["total"])
        same_date = (cand["date"] == best["date"]) and best["date"] is not None
        if same_store and same_total and same_date:
            consensus = True
            break

    source_tag = "consensus" if consensus else best["source"]

    # Fill missing fields from higher-priority candidates (tesseract -> ocr -> vision)
    sorted_candidates = sorted(candidates, key=lambda c: c["priority"])
    store = best["store"]
    total = best["total"]
    date_iso = best["date"]
    for cand in sorted_candidates:
        if store is None and cand["store"]:
            store = cand["store"]
        if total is None and cand["total"] is not None:
            total = cand["total"]
        if date_iso is None and cand["date"]:
            date_iso = cand["date"]

    return store, total, date_iso, source_tag

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
        "vision": GCV_ENABLED,
        "paddle_vl": PADDLE_VL_ENABLED,
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
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id in token")

    fname = file.filename or "unknown.jpg"
    tmp = DATA / "raw_images" / f"{uuid.uuid4().hex}_{fname}"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    tmp.write_bytes(content)

    job = await _register_job(user_id=user_id, filename=fname)
    asyncio.create_task(_process_job(job["id"], tmp, user_id, fname))
    return _public_job(job)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user=Depends(get_current_user)):
    job = await _get_job(job_id)
    if not job or job.get("user_id") != user.get("sub"):
        raise HTTPException(status_code=404, detail="Job not found")
    return _public_job(job)


async def _process_job(job_id: str, tmp_path: Path, user_id: str, filename: str):
    await _update_job(job_id, status="processing", started_at=time.time())
    try:
        result = await _process_receipt_pipeline(tmp_path, user_id, filename)
        await _update_job(
            job_id,
            status="completed",
            result=result,
            finished_at=time.time(),
        )
    except Exception as exc:
        logging.exception("Failed to process receipt job %s", job_id)
        await _update_job(
            job_id,
            status="failed",
            error=str(exc),
            finished_at=time.time(),
        )


async def _process_receipt_pipeline(tmp_path: Path, user_id: str, filename: str) -> dict:
    tmp = Path(tmp_path)
    if not tmp.exists():
        raise FileNotFoundError(f"Uploaded file missing: {tmp}")

    content = tmp.read_bytes()
    fname = filename

    img = None
    try:
        img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(str(tmp))

    fields = []
    yolo_store = yolo_total = yolo_date = None
    yolo_total_text = None
    yolo_total_attempts: list[str] = []
    if img is not None:
        try:
            fields = detect_fields(img)
            if fields:
                best = max(
                    [f for f in fields if f["name"] == "Merchant"],
                    key=lambda x: x["conf"],
                    default=None,
                )
                if best:
                    yolo_store = ocr_crop(img, best["box"], psm=7)

                best = max(
                    [f for f in fields if f["name"] == "Total"],
                    key=lambda x: x["conf"],
                    default=None,
                )
                if best:
                    yolo_total_val, yolo_text, tries = ocr_amount_from_crop(img, best["box"])
                    if yolo_total_val is not None:
                        yolo_total = yolo_total_val
                        yolo_total_text = yolo_text
                    if tries:
                        yolo_total_attempts = tries

                best = max(
                    [f for f in fields if f["name"] == "Date"],
                    key=lambda x: x["conf"],
                    default=None,
                )
                if best:
                    date_txt = ocr_crop(img, best["box"], psm=6)
                    yolo_date = extract_date(date_txt)
        except Exception:
            fields = []

    paddle_task = None
    tess_task = None
    ocr_space_task = None

    if PADDLE_VL_ENABLED:
        paddle_task = asyncio.create_task(
            OCR_STRATEGY_CONTEXT.run("paddle_vl", image_bytes=content, filename=fname)
        )
    if TESSERACT_ENABLED and OCR_STRATEGY_CONTEXT.has("tesseract"):
        tess_task = asyncio.create_task(
            OCR_STRATEGY_CONTEXT.run("tesseract", image_bytes=content, filename=fname)
        )
    if OCR_SPACE_ENABLED:
        ocr_space_task = asyncio.create_task(
            OCR_STRATEGY_CONTEXT.run("ocr_space", image_bytes=content, filename=fname)
        )

    vl_store = vl_total = vl_date = None
    vl_source_tag = None
    vl_used = False
    vl_payload = {"ok": False, "text": "", "error": None}
    vl_timeout = False
    if paddle_task:
        try:
            vl_result = await asyncio.wait_for(paddle_task, timeout=60)
            vl_payload = vl_result.payload
            if vl_payload.get("ok") and vl_payload.get("text"):
                s, t, d = parse_fields(vl_payload["text"])
                vl_store, vl_total, vl_date = s, t, d
                vl_source_tag = vl_result.name
                vl_used = True
        except asyncio.TimeoutError:
            paddle_task.cancel()
            vl_timeout = True
        except Exception:
            logging.exception("PaddleOCR-VL processing failed")

    if vl_payload.get("ok") and fields:
        for det in fields:
            crop_bytes = _crop_to_bytes(img, det["box"]) if img is not None else None
            if not crop_bytes:
                continue
            prompt = None
            if det["name"] == "Merchant":
                prompt = "Extract the store or merchant name from this receipt snippet."
            elif det["name"] == "Total":
                prompt = "Extract the total amount the customer needs to pay. Return only the numeric amount with currency if present."
            elif det["name"] == "Date":
                prompt = "Extract the transaction or receipt date in YYYY-MM-DD format if possible."
            crop_payload = paddle_vl_text(crop_bytes, prompt=prompt)
            if not crop_payload.get("ok") or not crop_payload.get("text"):
                continue
            cs, ct, cd = parse_fields(crop_payload["text"])
            if det["name"] == "Merchant" and cs and not vl_store:
                vl_store = cs
            elif det["name"] == "Total" and ct is not None and vl_total is None:
                vl_total = ct
            elif det["name"] == "Date" and cd and not vl_date:
                vl_date = cd
            vl_used = True

    rec = {"text": "", "mean_conf": 0.0, "words": []}
    tess_text = ""
    tess_conf = 0.0
    if tess_task:
        try:
            tess_result = await tess_task
            rec = tess_result.payload
            tess_text = rec.get("text", "") or ""
            tess_conf = rec.get("mean_conf", 0.0)
        except Exception:
            logging.exception("Tesseract processing failed")

    store_t = total_t = date_t = None
    if tess_text:
        store_t, total_t, date_t = parse_fields_from_ocr(rec)

    ocrs = {"ok": False, "text": "", "raw": None, "error": None, "http": None}
    if ocr_space_task:
        try:
            ocr_space_result = await ocr_space_task
            ocrs = ocr_space_result.payload
        except Exception:
            logging.exception("OCR.space processing failed")

    vision_used = False
    vision_res = {"ok": False, "text": "", "error": None, "info": None}
    if GCV_ENABLED:
        need_vision = False
        if store_t is None or total_t is None or date_t is None:
            need_vision = True
        if vl_store is None or vl_total is None or vl_date is None:
            need_vision = True
        if (tess_conf or 0) < 55:
            need_vision = True
        if need_vision:
            vision_res = (
                await OCR_STRATEGY_CONTEXT.run(
                    "google_vision", image_bytes=content, filename=fname
                )
            ).payload
            vision_used = True

    paddle_candidate = None
    if any(v is not None for v in (vl_store, vl_total, vl_date)):
        paddle_candidate = {
            "store": vl_store,
            "total": vl_total,
            "date": vl_date,
            "confidence": 90.0 if vl_payload.get("ok") else 0.0,
        }
    store_r, total_r, date_r, source_tag = resolve_fields(
        rec, tess_conf, ocrs, vision_res, paddle=paddle_candidate
    )

    store = yolo_store or vl_store or store_r
    total = (
        yolo_total
        if yolo_total is not None
        else (vl_total if vl_total is not None else total_r)
    )
    date_iso = yolo_date or vl_date or date_r

    store_norm = normalize_store_name(store) if store else None

    canon, canon_cat, canon_score = correct_store_name(store_norm or store)
    if canon and canon_score and canon_score >= 0.86:
        store = canon
        store_norm = canon

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

    if store == vl_store or total == vl_total or date_iso == vl_date:
        source_tag = vl_source_tag or source_tag

    ocr_space_used = bool(ocrs.get("ok"))
    source_labels = {
        "paddle_vl": "PaddleOCR-VL",
        "tesseract": "Tesseract",
        "ocr_space": "OCR.space",
        "vision": "Google Vision",
        "consensus": "Consensus (multiple engines agreed)",
    }
    source_friendly = source_labels.get(source_tag, source_tag)

    # If Tesseract confidence is low, rely on Google Vision fields if present
    final_store = store
    final_total = total
    final_date = date_iso
    if (tess_conf or 0) < 50 and vision_res.get("ok") and vision_res.get("text"):
        gv_store, gv_total, gv_date = parse_fields(vision_res["text"])
        if gv_store or gv_total is not None or gv_date:
            final_store = gv_store or final_store
            final_total = gv_total if gv_total is not None else final_total
            final_date = gv_date or final_date
            source_tag = "vision"

    insert_receipt(
        _id=tmp.stem,
        user_id=user_id,
        store=final_store,
        store_norm=normalize_store_name(final_store) if final_store else None,
        date_iso=final_date,
        total=_clean_num(final_total),
        category=category,
        category_source=source,
        confidence=_clean_num(confidence),
        ocr_conf=_clean_num(rec.get("mean_conf")),
        text=tess_text,
    )

    return {
        "id": tmp.stem,
        "store": final_store,
        "store_normalized": normalize_store_name(final_store) if final_store else None,
        "date": final_date,
        "total": final_total,
        "yolo_total_text": yolo_total_text,
        "yolo_total_attempts": yolo_total_attempts,
        "category": category,
        "confidence": confidence,
        "category_source": source,
        "reason": reason,
        "text": tess_text,
        "ocr_conf": rec.get("mean_conf"),
        "yolo_used": bool(fields),
        "ocr_space_used": ocr_space_used,
        "ocr_space_ok": ocr_space_used,
        "ocr_space_http": ocrs.get("http"),
        "ocr_space_err": ocrs.get("error"),
        "paddle_vl_used": vl_used,
        "paddle_vl_ok": vl_payload.get("ok", False),
        "paddle_vl_err": vl_payload.get("error"),
        "paddle_vl_timeout": vl_timeout,
        "vision_used": vision_used,
        "vision_ok": vision_res.get("ok", False),
        "vision_err": vision_res.get("error"),
        "vision_conf": (vision_res.get("info") or {}).get("confidence")
        if isinstance(vision_res.get("info"), dict)
        else None,
        "ocr_source": source_tag,
        "ocr_source_label": source_friendly,
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


@app.get("/stats/top_merchants")
def get_top_merchants(limit: int = 5, user=Depends(get_current_user)):
    return top_merchants_current_month(user_id=user.get("sub"), limit=limit)


@app.get("/stats/weekday_spend")
def get_weekday_spend(user=Depends(get_current_user)):
    return weekday_spend(user_id=user.get("sub"))


@app.get("/stats/rolling_30")
def get_rolling_30(user=Depends(get_current_user)):
    return rolling_30_day_spend(user_id=user.get("sub"))


@app.get("/receipts/low_confidence")
def get_low_confidence(threshold: float = 0.6, limit: int = 50, user=Depends(get_current_user)):
    return low_confidence_receipts(user_id=user.get("sub"), threshold=threshold, limit=limit)

@app.patch("/receipts/{rid}")
def update_receipt(rid: str, upd: ReceiptUpdate, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")

    with SessionLocal() as db:
        r = (
            db.query(Receipt)
            .filter(Receipt.id == rid, Receipt.user_id == user_id)
            .first()
        )
        if not r:
            raise HTTPException(status_code=404, detail="not found")

        original_values = {
            "store": r.store,
            "date": r.date,
            "total": r.total,
            "category": r.category,
        }

        data = upd.model_dump(exclude_unset=True)
        if "date" in data and data["date"] is not None:
            try:
                data["date"] = _date.fromisoformat(data["date"])
            except Exception:
                data["date"] = None

        corrections: list[ReceiptCorrection] = []
        for field in ("store", "date", "total", "category"):
            if field in data:
                old_val = original_values.get(field)
                new_val = data[field]
                if _serialize_value(old_val) != _serialize_value(new_val):
                    corrections.append(
                        ReceiptCorrection(
                            receipt_id=rid,
                            user_id=user_id,
                            field_name=field,
                            old_value=_serialize_value(old_val),
                            new_value=_serialize_value(new_val),
                            change_type="ocr"
                            if field in {"store", "date", "total"}
                            else "category",
                        )
                    )

        for k, v in data.items():
            setattr(r, k, v)

        if corrections:
            db.add_all(corrections)

        db.commit()

    # Track usage of custom labels (if category changed to a custom one)
    if "category" in data and data["category"]:
        new_cat = data["category"]
        if new_cat not in CATS_PUBLIC:
            # It's a custom label - increment usage
            increment_label_usage(user_id, new_cat)

    return {"ok": True}


@app.get("/logs/corrections")
def get_correction_logs(limit: int = 200, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")

    with SessionLocal() as db:
        rows = (
            db.query(ReceiptCorrection)
            .filter(ReceiptCorrection.user_id == user_id)
            .order_by(ReceiptCorrection.logged_at.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "id": row.id,
            "receipt_id": row.receipt_id,
            "field": row.field_name,
            "old": row.old_value,
            "new": row.new_value,
            "type": row.change_type,
            "logged_at": row.logged_at.isoformat() if row.logged_at else None,
        }
        for row in rows
    ]
@app.get("/debug/token")
async def debug_token(request: Request):
    auth = request.headers.get("authorization")
    return {"auth_header": auth}


# ================== Custom Labels API ==================

class CustomLabelCreate(BaseModel):
    name: str
    color: str | None = None
    icon: str | None = None
    description: str | None = None

class CustomLabelUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    icon: str | None = None
    description: str | None = None


@app.get("/custom_labels")
def get_custom_labels(user=Depends(get_current_user)):
    """List all custom labels for the current user."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    return list_custom_labels(user_id)


@app.post("/custom_labels")
def create_label(data: CustomLabelCreate, user=Depends(get_current_user)):
    """Create a new custom label."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    
    if not data.name or not data.name.strip():
        raise HTTPException(status_code=400, detail="Label name is required")
    
    try:
        label = create_custom_label(
            user_id=user_id,
            name=data.name.strip(),
            color=data.color,
            icon=data.icon,
            description=data.description,
        )
        return {"ok": True, "label": label}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/custom_labels/{label_id}")
def get_label(label_id: str, user=Depends(get_current_user)):
    """Get a single custom label by ID."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    
    label = get_custom_label(user_id, label_id)
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.patch("/custom_labels/{label_id}")
def patch_label(label_id: str, data: CustomLabelUpdate, user=Depends(get_current_user)):
    """Update a custom label."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    
    try:
        label = update_custom_label(
            user_id=user_id,
            label_id=label_id,
            name=data.name.strip() if data.name else None,
            color=data.color,
            icon=data.icon,
            description=data.description,
        )
        if not label:
            raise HTTPException(status_code=404, detail="Label not found")
        return {"ok": True, "label": label}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.delete("/custom_labels/{label_id}")
def remove_label(label_id: str, user=Depends(get_current_user)):
    """Delete a custom label."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    
    deleted = delete_custom_label(user_id, label_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Label not found")
    return {"ok": True}


@app.get("/categories")
def get_all_categories(user=Depends(get_current_user)):
    """
    Get all available categories: built-in + user's custom labels.
    Useful for populating category dropdowns.
    """
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user id")
    
    # Built-in categories
    builtin = [
        {"name": cat, "type": "builtin", "color": None, "icon": None}
        for cat in CATS_PUBLIC
    ]
    
    # Custom labels
    custom = list_custom_labels(user_id)
    custom_cats = [
        {
            "name": label["name"],
            "type": "custom",
            "color": label.get("color"),
            "icon": label.get("icon"),
            "id": label["id"],
            "usage_count": label.get("usage_count", 0),
        }
        for label in custom
    ]
    
    return {"builtin": builtin, "custom": custom_cats}


# Ensure DB tables exist on import
init_db()

