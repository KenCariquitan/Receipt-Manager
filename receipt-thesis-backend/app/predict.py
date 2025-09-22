import sys
from joblib import load
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1]/"models"
vec = load(MODELS/"vectorizer.joblib")
clf = load(MODELS/"classifier.joblib")

text = sys.stdin.read()
print(clf.predict(vec.transform([text]))[0])
