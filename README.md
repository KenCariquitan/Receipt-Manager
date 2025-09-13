# Receipt Thesis — Backend Only (Beginner Friendly)

This folder gives you a ready-to-run backend that:
1) Accepts a **receipt image** upload.
2) Uses **Tesseract OCR** to extract text.
3) Parses **store name**, **total amount**, and **date**.
4) Classifies the receipt into **Utilities / Food / Transportation / Health & Wellness / Others** using:
   - **PH‑specific rules** (Meralco, PLDT, Jollibee, etc.) and keywords, then
   - an ML model (after you train it).
5) Collects **feedback** so the model can learn from your corrections.

> Flutter mobile app will come later. You can test everything right now in a browser.

---

## Quick Start (Windows/macOS/Linux)

### 0) Install system Tesseract (OCR engine)
- **Windows:** Install from https://github.com/UB-Mannheim/tesseract/wiki
  - If needed, open `app/ocr.py` and set the `pytesseract` path (instructions inside).
- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`

### 1) Create & activate a Python environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install Python packages
```bash
pip install -r requirements.txt
```

### 3) Run the API (backend)
```bash
uvicorn app.api:app --reload --port 8000
```
Open your browser to:
- **Health check:** http://localhost:8000/health
- **Interactive API docs:**   

### 4) Test without coding
In http://localhost:8000/docs
- Use **POST /upload_receipt** → upload an image (JPG/PNG). You’ll see store/total/date and a category.
- If no model is trained yet, the category may come from **rules** (e.g., “brand‑match: MERALCO”).

### 5) (Optional) Build a dataset and train the model
1. Put several images into: `data/raw_images/`
2. Run dataset builder:
   ```bash
   python train/build_dataset.py
   ```
3. Label the receipts quickly (one‑time setup):
   ```bash
   streamlit run labeler/label_app.py
   ```
   - A `data/labels.csv` file will be created while you label.
4. Merge + train:
   ```bash
   python train/build_dataset.py   # merges labels → dataset.csv
   python train/train.py           # trains ML, saves models/vectorizer.joblib + classifier.joblib
   ```
5. **Restart** the API and upload again. Now predictions may come from **source: "ml"** with a probability.

### 6) Feedback loop (learning from corrections)
- In `/upload_receipt` output, copy the OCR text and send it with your correct label:
  ```bash
  curl -X POST http://localhost:8000/feedback -d "text=PASTE_OCR_TEXT_HERE" -d "true_label=Food"
  ```
- After collecting some corrections:
  ```bash
  curl -X POST http://localhost:8000/retrain_incremental
  ```

### 7) Dashboard (optional)
```bash
streamlit run dashboard/dashboard.py
```

---

## Folder Map

```
.
├─ data/
│  ├─ raw_images/           # put receipt images here
│  ├─ ocr_raw.csv           # OCR text + meta
│  ├─ parsed_fields.csv     # store / total / date parsed
│  ├─ labels.csv            # your manual labels (created by labeler)
│  ├─ dataset.csv           # merged text + label (train set)
│  ├─ feedback.csv          # user corrections
├─ models/
│  ├─ vectorizer.joblib     # saved after training
│  ├─ classifier.joblib     # saved after training
├─ app/
│  ├─ api.py                # FastAPI (upload, classify, feedback, retrain)
│  ├─ ocr.py                # Tesseract OCR + preprocessing
│  ├─ parser.py             # extract store / total / date
│  ├─ ph_rules.py           # PH brand/keyword categorization
│  ├─ predict.py            # classify raw text from stdin (optional)
├─ train/
│  ├─ build_dataset.py      # OCR → parse → (merge labels) → dataset.csv
│  ├─ train.py              # train ML model (TF‑IDF + SGD)
│  ├─ evaluate.py           # evaluate model on dataset.csv
├─ labeler/
│  ├─ label_app.py          # tiny Streamlit app for labeling
├─ dashboard/
│  ├─ dashboard.py          # simple Streamlit dashboard
├─ requirements.txt
└─ README.md
```

If you get stuck anywhere, copy the error text and I’ll fix it with you. 😊
