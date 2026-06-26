import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Batch Prediction")
st.caption("Unggah file CSV untuk prediksi massal")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_resource
def load_config():
    with open("model_info.json", "r") as f:
        return json.load(f)

model = load_model()
config = load_config()
features = config["fitur"]
thresholds = config["thresholds_rbds"]

# ============================================================
# FUNGSI
# ============================================================
def rbds_diagnose(sample: dict) -> list:
    anomalies = []
    for param, rules in thresholds.items():
        val = sample.get(param)
        if val is None:
            continue
        label = rules["label"]
        unit = rules["unit"]

        if rules["min"] is not None and val < rules["min"]:
            anomalies.append({
                "parameter": label,
                "nilai": round(val, 4),
                "batas": f">= {rules['min']} {unit}".strip(),
                "status": "DI BAWAH BATAS"
            })
        if rules["max"] is not None and val > rules["max"]:
            anomalies.append({
                "parameter": label,
                "nilai": round(val, 4),
                "batas": f"<= {rules['max']} {unit}".strip(),
                "status": "MELEBIHI BATAS"
            })
    return anomalies

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    X = df[features]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = df.copy()
    results["Prediksi"] = predictions.map({1: "LAYAK", 0: "TIDAK LAYAK"})
    results["Prob_Layak"] = probabilities[:, 1]
    results["Prob_Tidak_Layak"] = probabilities[:, 0]
    results["Jumlah_Anomali"] = 0
    results["Detail_Anomali"] = ""
    
    for idx, row in results.iterrows():
        if predictions[idx] == 0:
            sample = row[features].to_dict()
            anomalies = rbds_diagnose(sample)
            results.at[idx, "Jumlah_Anomali"] = len(anomalies)
            if anomalies:
                detail = "; ".join([f"{a['parameter']}: {a['nilai']} ({a['status']})" for a in anomalies])
                results.at[idx, "Detail_Anomali"] = detail
    
    return results

# ============================================================
# UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Kolom tidak ditemukan: {missing_cols}")
        st.stop()
    
    st.success(f"✅ {len(df)} sampel ditemukan")
    
    if st.button("🚀 Proses Prediksi", type="primary"):
        with st.spinner("Memproses..."):
            results = predict_batch(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sampel", len(results))
        with col2:
            st.metric("✅ Layak", results[results["Prediksi"] == "LAYAK"].shape[0])
        with col3:
            st.metric("❌ Tidak Layak", results[results["Prediksi"] == "TIDAK LAYAK"].shape[0])
        
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        csv = results.to_csv(index=False)
        st.download_button(
            "📥 Download Hasil (CSV)",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
else:
    st.info("📂 Upload file CSV untuk memulai")
