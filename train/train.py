import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
MODELS = ROOT/"models"
MODELS.mkdir(exist_ok=True, parents=True)

CATS = ["Utilities","Food","Transportation","Health & Wellness","Others"]

df = pd.read_csv(DATA/"dataset.csv").dropna(subset=["text","label"]).sample(frac=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.95)),
    ("clf", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred, labels=CATS))

dump(pipe.named_steps["tfidf"], MODELS/"vectorizer.joblib")
dump(pipe.named_steps["clf"], MODELS/"classifier.joblib")
print("Saved models/")
