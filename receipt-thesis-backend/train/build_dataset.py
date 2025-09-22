import pandas as pd
from pathlib import Path
from tqdm import tqdm
from app.ocr import ocr_image_path
from app.parser import parse_fields

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
RAW = DATA/"raw_images"
DATA.mkdir(exist_ok=True, parents=True)

# 1) OCR all images
rows = []
for p in tqdm(sorted(RAW.glob("*"))):
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".webp", ".bmp"}:
        continue
    rec = ocr_image_path(str(p))
    rec["id"] = p.stem
    rows.append(rec)

ocr = pd.DataFrame(rows)
ocr.to_csv(DATA/"ocr_raw.csv", index=False)

# 2) Parse fields
stores, totals, dates = [], [], []
for t in ocr.text.fillna(""):
    s, tot, d = parse_fields(t)
    stores.append(s); totals.append(tot); dates.append(d)
ocr["store"], ocr["total"], ocr["date"] = stores, totals, dates
ocr.to_csv(DATA/"parsed_fields.csv", index=False)
print("Wrote parsed_fields.csv")

# 3) Merge with labels (if exists) to make dataset
labels_path = DATA/"labels.csv"
if labels_path.exists():
    labels = pd.read_csv(labels_path)
    merged = ocr.merge(labels, on="id")
    merged[["id","text","label"]].to_csv(DATA/"dataset.csv", index=False)
    print("Wrote dataset.csv")
else:
    print("labels.csv not found yet — run labeler to create labels.")
