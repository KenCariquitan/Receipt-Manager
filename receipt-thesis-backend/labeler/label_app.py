import streamlit as st
import pandas as pd
from pathlib import Path

CATS = ["Utilities","Food","Transportation","Health & Wellness","Groceries","Others"]

st.title("Receipt Labeler")
DATA = Path(__file__).resolve().parents[1]/"data"
ocr = pd.read_csv(DATA/"parsed_fields.csv") if (DATA/"parsed_fields.csv").exists() else pd.read_csv(DATA/"ocr_raw.csv")
labels_path = DATA/"labels.csv"
labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame(columns=["id","label"])
lab_map = dict(zip(labels.id.astype(str), labels.label))

for i, row in ocr.iterrows():
    rid = str(row.get("id", i))
    st.subheader(rid)
    with st.expander("OCR Text"):
        st.text(row.get("text", ""))
    st.write({"store": row.get("store"), "total": row.get("total"), "date": row.get("date")})
    cur = lab_map.get(rid, "Others")
    choice = st.radio("Label", CATS, index=CATS.index(cur), key=f"lab_{rid}")
    if st.button("Save", key=f"save_{rid}"):
        lab_map[rid] = choice
        out = pd.DataFrame({"id": list(lab_map.keys()), "label": list(lab_map.values())})
        out.to_csv(labels_path, index=False)
        st.success("Saved!")
