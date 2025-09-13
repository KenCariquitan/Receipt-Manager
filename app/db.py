

from __future__ import annotations
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Iterable
from sqlalchemy import (create_engine, Column, String, Float, Date, DateTime, Text, Index)
from sqlalchemy.orm import declarative_base, sessionmaker

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "receipts.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(String, primary_key=True)      # your upload id
    store = Column(String, nullable=True)
    store_normalized = Column(String, nullable=True)
    date = Column(Date, nullable=True)         # ISO date
    total = Column(Float, nullable=True)
    category = Column(String, nullable=True)   # Utilities/Food/Groceries/...
    category_source = Column(String, nullable=True)  # "rule" | "ml"
    confidence = Column(Float, nullable=True)  # ML probability (if any)
    ocr_conf = Column(Float, nullable=True)    # mean OCR confidence
    text = Column(Text, nullable=True)         # full OCR text (optional)
    created_at = Column(DateTime, default=datetime.utcnow)

# helpful indexes for queries
Index("idx_receipts_date", Receipt.date)
Index("idx_receipts_category", Receipt.category)
Index("idx_receipts_created", Receipt.created_at)

def init_db():
    Base.metadata.create_all(bind=engine)

def insert_receipt(
    _id: str, store: Optional[str], store_norm: Optional[str],
    date_iso: Optional[str], total: Optional[float],
    category: Optional[str], category_source: Optional[str],
    confidence: Optional[float], ocr_conf: Optional[float],
    text: Optional[str]
):
    with SessionLocal() as db:
        d: Optional[date] = None
        if date_iso:
            try:
                d = date.fromisoformat(date_iso)
            except Exception:
                d = None
        r = Receipt(
            id=_id, store=store, store_normalized=store_norm,
            date=d, total=total, category=category, category_source=category_source,
            confidence=confidence, ocr_conf=ocr_conf, text=text
        )
        db.merge(r)  # upsert by id (safe if reprocessing same image)
        db.commit()

def list_receipts(limit: int = 50, offset: int = 0) -> list[Receipt]:
    with SessionLocal() as db:
        return db.query(Receipt).order_by(Receipt.created_at.desc()).offset(offset).limit(limit).all()

def stats_by_category() -> list[dict]:
    from sqlalchemy import func
    with SessionLocal() as db:
        rows = db.query(
            Receipt.category,
            func.count(Receipt.id),
            func.coalesce(func.sum(Receipt.total), 0.0)
        ).group_by(Receipt.category).all()
        return [{"category": c or "Unknown", "count": int(n), "total": float(t)} for c,n,t in rows]

def stats_by_month(year: int) -> list[dict]:
    from sqlalchemy import func
    with SessionLocal() as db:
        rows = db.query(
            func.strftime("%Y-%m", Receipt.date).label("ym"),
            func.coalesce(func.sum(Receipt.total), 0.0),
            func.count(Receipt.id)
        ).filter(
            Receipt.date.isnot(None),
            func.strftime("%Y", Receipt.date) == str(year)
        ).group_by("ym").order_by("ym").all()
        return [{"month": ym, "total": float(t), "count": int(n)} for ym,t,n in rows]

def stats_summary() -> dict:
    from sqlalchemy import func
    today = datetime.utcnow().date()
    first_of_month = today.replace(day=1)
    with SessionLocal() as db:
        total_spend = db.query(func.coalesce(func.sum(Receipt.total), 0.0)).scalar() or 0.0
        total_receipts = db.query(func.count(Receipt.id)).scalar() or 0
        mtd_spend = db.query(func.coalesce(func.sum(Receipt.total), 0.0))\
                      .filter(Receipt.date >= first_of_month).scalar() or 0.0
        top = db.query(Receipt.category, func.coalesce(func.sum(Receipt.total),0.0))\
                .group_by(Receipt.category).order_by(func.sum(Receipt.total).desc()).first()
        return {
            "total_spend": float(total_spend),
            "total_receipts": int(total_receipts),
            "month_to_date_spend": float(mtd_spend),
            "top_category": top[0] if top else None,
            "top_category_total": float(top[1]) if top else 0.0
        }
