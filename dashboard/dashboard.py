import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Receipt Dashboard")
DATA = Path(__file__).resolve().parents[1]/"data"
parsed = DATA/"parsed_fields.csv"
if parsed.exists():
    df = pd.read_csv(parsed)
    st.dataframe(df[["id","store","date","total"]].head(100))
    if "category" in df:
        st.bar_chart(df["category"].value_counts())
else:
    st.info("No parsed_fields.csv yet. Run train/build_dataset.py")
