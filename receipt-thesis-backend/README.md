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

py -3.11 -m venv .venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
#gitbash
source .venv/Scripts/activate
```

### 2) Install Python packages
```bash
pip install -r requirements.txt
```

### 3) Run the API (backend)
```bash
uvicorn app.api:app --reload --port 8000
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Open your browser to:
- **Health check:** http://localhost:8000/health
- **Interactive API docs:**   

### 4) Test without coding
In http://localhost:8000/docs
- Use **POST /upload_receipt** → upload an image (JPG/PNG). You’ll see store/total/date and a category.
- If no model is trained yet, the category may come from **rules** (e.g., “brand‑match: MERALCO”).
