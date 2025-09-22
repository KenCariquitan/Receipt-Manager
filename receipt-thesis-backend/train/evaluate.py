from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

CATS = ["Utilities", "Food", "Groceries", "Transportation", "Health & Wellness", "Others"]
CAT_SET = set(CATS)

csv_path = DATA / "dataset.csv"
assert csv_path.exists(), f"dataset.csv not found at {csv_path}"
df = pd.read_csv(csv_path)

# Build text like in train.py (must be consistent!)
def make_text(row):
    parts = []
    for col in ("text", "store", "date"):
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    if "total" in row and pd.notna(row["total"]):
        try:
            amt = float(row["total"])
            parts.append(f"TOTALPHP_{int(round(amt))}")
        except Exception:
            pass
    return " ".join(parts).strip()

if "text" in df.columns:
    X_text = df.apply(lambda r: make_text(r), axis=1)
else:
    X_text = df.apply(lambda r: make_text(r), axis=1)

if "label" in df.columns:
    y = df["label"].fillna("Others").astype(str)
elif "category" in df.columns:
    y = df["category"].fillna("Others").astype(str)
else:
    raise ValueError("No 'label' or 'category' column found in dataset.csv")

y = y.apply(lambda c: c if c in CAT_SET else "Others")

# Drop empties
mask_nonempty = X_text.str.len() > 0
X_text = X_text[mask_nonempty]
y = y[mask_nonempty]

# Load models
vec = load(MODELS / "vectorizer.joblib")
clf = load(MODELS / "classifier.joblib")

X = vec.transform(X_text)
y_pred = clf.predict(X)

print("\n=== Full-dataset evaluation ===")
print(classification_report(y, y_pred, labels=CATS, digits=3))
print("\n=== Confusion matrix (rows=true, cols=pred) ===")
print(confusion_matrix(y, y_pred, labels=CATS))

print("\nMacro-F1:", round(float(f1_score(y, y_pred, average='macro')), 4))
