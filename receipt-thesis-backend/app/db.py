# app/db.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Iterable, Any

from sqlalchemy import (
    create_engine, Column, String, Float, Date, DateTime, Text, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()  # pick up SUPABASE_DB_URL, etc.

# --- Choose DB based on env ---
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
if SUPABASE_DB_URL:
    # Example (pooler): postgres://USER:PASSWORD@HOST:6543/postgres?sslmode=require
    engine = create_engine(
        SUPABASE_DB_URL,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"} if "sslmode" not in SUPABASE_DB_URL else {},
    )
    DB_DESC = "Supabase Postgres"
else:
    # Local fallback (dev): SQLite in ./data/receipts.db
    DATA = Path(__file__).resolve().parents[1] / "data"
    DATA.mkdir(parents=True, exist_ok=True)
    DB_PATH = DATA / "receipts.db"
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    DB_DESC = "SQLite (local)"

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(String, primary_key=True)             # upload id
    user_id = Column(String, index=True, nullable=True)  # Supabase auth uid (uuid string)
    store = Column(String, nullable=True)
    store_normalized = Column(String, nullable=True)
    date = Column(Date, nullable=True)                # ISO date
    total = Column(Float, nullable=True)
    category = Column(String, nullable=True)          # Utilities/Food/Groceries/...
    category_source = Column(String, nullable=True)   # "rule" | "ml"
    confidence = Column(Float, nullable=True)         # ML probability
    ocr_conf = Column(Float, nullable=True)           # OCR mean confidence
    text = Column(Text, nullable=True)                # full OCR text (optional)
    created_at = Column(DateTime, default=datetime.utcnow)

Index("idx_receipts_user_created", Receipt.user_id, Receipt.created_at)
Index("idx_receipts_user_date", Receipt.user_id, Receipt.date)
Index("idx_receipts_user_category", Receipt.user_id, Receipt.category)


def init_db():
    Base.metadata.create_all(bind=engine)


def insert_receipt(
    _id: str,
    user_id: str,
    store: Optional[str],
    store_norm: Optional[str],
    date_iso: Optional[str],
    total: Optional[float],
    category: Optional[str],
    category_source: Optional[str],
    confidence: Optional[float],
    ocr_conf: Optional[float],
    text: Optional[str],
):
    def _clean_num(x):
        from math import isnan, isinf
        try:
            return None if x is None or isnan(x) or isinf(x) else float(x)
        except Exception:
            return None

    with SessionLocal() as db:
        d: Optional[date] = None
        if date_iso:
            try:
                d = date.fromisoformat(date_iso)
            except Exception:
                d = None
        r = Receipt(
            id=_id,
            user_id=user_id,
            store=store,
            store_normalized=store_norm,
            date=d,
            total=_clean_num(total),
            category=category,
            category_source=category_source,
            confidence=_clean_num(confidence),
            ocr_conf=_clean_num(ocr_conf),
            text=text,
        )
        db.merge(r)  # upsert by primary key
        db.commit()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def list_receipts(user_id: str, limit: int = 50, offset: int = 0) -> list[Receipt]:
    with SessionLocal() as db:
        return (
            db.query(Receipt)
            .filter(Receipt.user_id == user_id)
            .order_by(Receipt.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


def stats_by_category(user_id: str) -> list[dict]:
    from sqlalchemy import func
    with SessionLocal() as db:
        rows = (
            db.query(
                Receipt.category,
                func.count(Receipt.id),
                func.coalesce(func.sum(Receipt.total), 0.0),
            )
            .filter(Receipt.user_id == user_id)
            .group_by(Receipt.category)
            .all()
        )
        return [{"category": c or "Unknown", "count": int(n), "total": float(t)} for c, n, t in rows]


def stats_by_month(year: int, user_id: str) -> list[dict]:
    from sqlalchemy import func, text
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            rows = db.execute(
                text("""
                    SELECT to_char(date, 'YYYY-MM') AS ym,
                           COALESCE(SUM(total), 0) AS total,
                           COUNT(id) AS count
                    FROM receipts
                    WHERE user_id = :uid
                      AND date IS NOT NULL
                      AND EXTRACT(YEAR FROM date) = :year
                    GROUP BY ym
                    ORDER BY ym
                """),
                {"uid": user_id, "year": year},
            ).fetchall()
            return [{"month": r[0], "total": float(r[1]), "count": int(r[2])} for r in rows]
        else:
            rows = (
                db.query(
                    func.strftime("%Y-%m", Receipt.date).label("ym"),
                    func.coalesce(func.sum(Receipt.total), 0.0),
                    func.count(Receipt.id),
                )
                .filter(
                    Receipt.user_id == user_id,
                    Receipt.date.isnot(None),
                    func.strftime("%Y", Receipt.date) == str(year),
                )
                .group_by("ym")
                .order_by("ym")
                .all()
            )
            return [{"month": ym, "total": float(t), "count": int(n)} for ym, t, n in rows]


def stats_summary(user_id: str) -> dict:
    from sqlalchemy import func, text
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            total_spend = (
                db.execute(text("SELECT COALESCE(SUM(total),0) FROM receipts WHERE user_id = :uid"), {"uid": user_id})
                .scalar()
                or 0.0
            )
            total_receipts = (
                db.execute(text("SELECT COUNT(id) FROM receipts WHERE user_id = :uid"), {"uid": user_id})
                .scalar()
                or 0
            )
            mtd = (
                db.execute(
                    text("""
                        SELECT COALESCE(SUM(total),0) FROM receipts
                        WHERE user_id = :uid
                          AND date IS NOT NULL
                          AND date >= date_trunc('month', CURRENT_DATE)
                    """),
                    {"uid": user_id},
                ).scalar()
                or 0.0
            )
            top = db.execute(
                text("""
                    SELECT category, COALESCE(SUM(total),0) AS t
                    FROM receipts
                    WHERE user_id = :uid
                    GROUP BY category
                    ORDER BY t DESC NULLS LAST
                    LIMIT 1
                """),
                {"uid": user_id},
            ).fetchone()
            return {
                "total_spend": float(total_spend),
                "total_receipts": int(total_receipts),
                "month_to_date_spend": float(mtd),
                "top_category": top[0] if top else None,
                "top_category_total": float(top[1]) if top else 0.0,
            }
        else:
            total_spend = (
                db.query(func.coalesce(func.sum(Receipt.total), 0.0))
                .filter(Receipt.user_id == user_id)
                .scalar()
                or 0.0
            )
            total_receipts = db.query(Receipt).filter(Receipt.user_id == user_id).count()
            from datetime import date as _d
            today = _d.today()
            first = today.replace(day=1)
            mtd_spend = (
                db.query(func.coalesce(func.sum(Receipt.total), 0.0))
                .filter(Receipt.user_id == user_id, Receipt.date >= first)
                .scalar()
                or 0.0
            )
            top = (
                db.query(Receipt.category, func.coalesce(func.sum(Receipt.total), 0.0))
                .filter(Receipt.user_id == user_id)
                .group_by(Receipt.category)
                .order_by(func.sum(Receipt.total).desc())
                .first()
            )
            return {
                "total_spend": float(total_spend),
                "total_receipts": int(total_receipts),
                "month_to_date_spend": float(mtd_spend),
                "top_category": top[0] if top else None,
                "top_category_total": float(top[1]) if top else 0.0,
            }


def top_merchants_current_month(user_id: str, limit: int = 5) -> list[dict]:
    from sqlalchemy import func, text
    limit = max(1, min(limit, 25))
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            rows = db.execute(
                text(
                    """
                    SELECT COALESCE(store_normalized, store) AS store,
                           COUNT(id) AS receipt_count,
                           COALESCE(SUM(total), 0) AS total_spend
                    FROM receipts
                    WHERE user_id = :uid
                      AND date >= date_trunc('month', CURRENT_DATE)
                      AND store IS NOT NULL
                    GROUP BY store, store_normalized
                    ORDER BY total_spend DESC NULLS LAST
                    LIMIT :limit
                    """
                ),
                {"uid": user_id, "limit": limit},
            ).fetchall()
            return [
                {
                    "store": r[0] or "Unknown",
                    "receipt_count": int(r[1]),
                    "total_spend": float(r[2]),
                }
                for r in rows
            ]
        else:
            today = datetime.utcnow().date()
            first = today.replace(day=1)
            rows = (
                db.query(
                    func.coalesce(Receipt.store_normalized, Receipt.store).label("store"),
                    func.count(Receipt.id),
                    func.coalesce(func.sum(Receipt.total), 0.0),
                )
                .filter(
                    Receipt.user_id == user_id,
                    Receipt.date.isnot(None),
                    Receipt.date >= first,
                )
                .group_by("store")
                .order_by(func.sum(Receipt.total).desc())
                .limit(limit)
                .all()
            )
            return [
                {"store": store or "Unknown", "receipt_count": int(count), "total_spend": float(total)}
                for store, count, total in rows
            ]


def weekday_spend(user_id: str) -> list[dict]:
    from sqlalchemy import func, text
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            rows = db.execute(
                text(
                    """
                    SELECT EXTRACT(DOW FROM date) AS dow,
                           COALESCE(SUM(total), 0) AS total_spend,
                           COUNT(id) AS receipt_count
                    FROM receipts
                    WHERE user_id = :uid
                      AND date IS NOT NULL
                    GROUP BY dow
                    ORDER BY dow
                    """
                ),
                {"uid": user_id},
            ).fetchall()
            return [
                {"weekday": int(r[0]), "total_spend": float(r[1]), "receipt_count": int(r[2])}
                for r in rows
            ]
        else:
            rows = (
                db.query(
                    func.strftime("%w", Receipt.date).label("dow"),
                    func.coalesce(func.sum(Receipt.total), 0.0),
                    func.count(Receipt.id),
                )
                .filter(Receipt.user_id == user_id, Receipt.date.isnot(None))
                .group_by("dow")
                .order_by("dow")
                .all()
            )
            return [
                {"weekday": int(dow), "total_spend": float(total), "receipt_count": int(count)}
                for dow, total, count in rows
            ]


def rolling_30_day_spend(user_id: str) -> list[dict]:
    from sqlalchemy import func, text
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            rows = db.execute(
                text(
                    """
                    SELECT date::date AS day,
                           COALESCE(SUM(total), 0) AS total_spend,
                           COUNT(id) AS receipt_count
                    FROM receipts
                    WHERE user_id = :uid
                      AND date >= CURRENT_DATE - INTERVAL '29 day'
                    GROUP BY day
                    ORDER BY day
                    """
                ),
                {"uid": user_id},
            ).fetchall()
            return [
                {"date": r[0].isoformat(), "total_spend": float(r[1]), "receipt_count": int(r[2])}
                for r in rows
            ]
        else:
            from datetime import date as _d, timedelta
            start = _d.today() - timedelta(days=29)
            rows = (
                db.query(
                    Receipt.date.label("day"),
                    func.coalesce(func.sum(Receipt.total), 0.0),
                    func.count(Receipt.id),
                )
                .filter(
                    Receipt.user_id == user_id,
                    Receipt.date.isnot(None),
                    Receipt.date >= start,
                )
                .group_by("day")
                .order_by("day")
                .all()
            )
            return [
                {"date": day.isoformat() if day else None, "total_spend": float(total), "receipt_count": int(count)}
                for day, total, count in rows
            ]


def low_confidence_receipts(user_id: str, threshold: float = 0.6, limit: int = 50) -> list[dict]:
    from sqlalchemy import func, text
    limit = max(1, min(limit, 200))
    threshold = max(0.0, min(threshold, 1.0))
    with SessionLocal() as db:
        if SUPABASE_DB_URL:
            rows = db.execute(
                text(
                    """
                    SELECT id, store, store_normalized, date, total, category,
                           confidence, ocr_conf, created_at
                    FROM receipts
                    WHERE user_id = :uid
                      AND (confidence IS NULL OR confidence < :threshold)
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"uid": user_id, "threshold": threshold, "limit": limit},
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "store": r[1],
                    "store_normalized": r[2],
                    "date": r[3].isoformat() if r[3] else None,
                    "total": float(r[4]) if r[4] is not None else None,
                    "category": r[5],
                    "confidence": float(r[6]) if r[6] is not None else None,
                    "ocr_conf": float(r[7]) if r[7] is not None else None,
                    "created_at": r[8].isoformat() if r[8] else None,
                }
                for r in rows
            ]
        else:
            rows = (
                db.query(Receipt)
                .filter(
                    Receipt.user_id == user_id,
                    (Receipt.confidence.is_(None)) | (Receipt.confidence < threshold),
                )
                .order_by(Receipt.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "store": r.store,
                    "store_normalized": r.store_normalized,
                    "date": r.date.isoformat() if r.date is not None else None,
                    "total": _to_float(r.total),
                    "category": r.category,
                    "confidence": _to_float(r.confidence),
                    "ocr_conf": _to_float(r.ocr_conf),
                    "created_at": r.created_at.isoformat() if r.created_at is not None else None,
                }
                for r in rows
            ]
