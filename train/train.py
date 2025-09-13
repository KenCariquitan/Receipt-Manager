import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Target labels (now includes Groceries)
CATS = ["Utilities", "Food", "Groceries", "Transportation", "Health & Wellness", "Others"]
CAT_SET = set(CATS)

# Load dataset
csv_path = DATA / "dataset.csv"
assert csv_path.exists(), f"dataset.csv not found at {csv_path}"
df = pd.read_csv(csv_path)

# -------- Build text feature robustly --------
# Prefer 'text' column if present; otherwise synthesize from other fields
def make_text(row):
    parts = []
    for col in ("text", "store", "date"):
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    # Add total as a token to help the model learn price patterns
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

# Clean labels and map unknowns to Others
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

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorizer + Classifier
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,            # adjust smaller if your dataset is tiny
    max_df=0.95
)

Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

# Class weights (help with class imbalance)
classes = np.array(CATS)
# restrict to classes that actually appear in y to compute weights; default Others for missing
present = sorted(set(y_train))
weights = dict(zip(
    present,
    compute_class_weight(class_weight="balanced", classes=np.array(present), y=y_train)
))
class_weight = {c: float(weights.get(c, 1.0)) for c in CATS}

clf = SGDClassifier(
    loss="log_loss",       # probabilistic
    max_iter=2000,
    tol=1e-3,
    random_state=42,
    class_weight=class_weight
)

clf.fit(Xtr, y_train)

# Quick eval
y_pred = clf.predict(Xte)
print("\n=== Classification report (test set) ===")
print(classification_report(y_test, y_pred, labels=CATS, digits=3))
print("\n=== Confusion matrix (rows=true, cols=pred) ===")
print(confusion_matrix(y_test, y_pred, labels=CATS))

# Save artifacts compatible with your API
dump(vec, MODELS / "vectorizer.joblib")
dump(clf, MODELS / "classifier.joblib")
print("\nSaved models/vectorizer.joblib and models/classifier.joblib")
