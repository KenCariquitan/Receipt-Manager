import pandas as pd
from sklearn.metrics import classification_report, f1_score
from joblib import load
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
MODELS = ROOT/"models"

df = pd.read_csv(DATA/"dataset.csv")
vec = load(MODELS/"vectorizer.joblib")
clf = load(MODELS/"classifier.joblib")
X = vec.transform(df.text)
print(classification_report(df.label, clf.predict(X), digits=3))
print("Macro-F1:", f1_score(df.label, clf.predict(X), average='macro'))
