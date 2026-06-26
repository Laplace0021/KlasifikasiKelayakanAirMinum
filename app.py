import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

# ============================================================
# LOAD MODEL & KONFIGURASI
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model

@st.cache_resource
def load_config():
    with open("model_info.json", "r") as f:
        config = json.load(f)
    return config

model = load_model()
config = load_config()
features = config["fitur"]
thresholds = config["thresholds_rbds"]
best_model_name = config.get("nama_model", "Decision Tree")

# ============================================================
# FUNGSI RBDS
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

def predict_and_diagnose(sample: dict) -> dict:
    input_df = pd.DataFrame([sample])[features]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    result = {
        "prediksi": "LAYAK" if prediction == 1 else "TIDAK LAYAK",
        "label": int(prediction),
        "probabilitas_layak": round(float(prob[1]), 4),
        "probabilitas_tidak_layak": round(float(prob[0]), 4),
        "anomali": []
    }

    if prediction == 0:
        result["anomali"] = rbds_diagnose(sample)

    return result

# ============================================================
# HEADER
# ============================================================
st.title("💧 Klasifikasi Kelayakan Air Minum")
st.caption(f"🤖 Model: {best_model_name} | 📊 Mengacu pada standar WHO/EPA")

# ============================================================
# INPUT PARAMETER - 3 Kolom
# ============================================================
st.subheader("📊 Masukkan Parameter Air")

input_data = {}

default_values = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 330.0,
    "Conductivity": 400.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 66.0,
    "Turbidity": 4.0
}

col1, col2, col3 = st.columns(3)

with col1:
    input_data["ph"] = st.number_input(
        "pH", 
        value=default_values["ph"], 
        format="%.4f",
        help="Batas: 6.5 - 8.5"
    )
    input_data["Hardness"] = st.number_input(
        "Hardness (mg/L)", 
        value=default_values["Hardness"], 
        format="%.4f",
        help="Batas: <= 300 mg/L"
    )
    input_data["Solids"] = st.number_input(
        "Solids / TDS (mg/L)", 
        value=default_values["Solids"], 
        format="%.4f",
        help="Batas: <= 500 mg/L"
    )

with col2:
    input_data["Chloramines"] = st.number_input(
        "Chloramines (mg/L)", 
        value=default_values["Chloramines"], 
        format="%.4f",
        help="Batas: <= 4.0 mg/L"
    )
    input_data["Sulfate"] = st.number_input(
        "Sulfate (mg/L)", 
        value=default_values["Sulfate"], 
        format="%.4f",
        help="Batas: <= 250 mg/L"
    )
    input_data["Conductivity"] = st.number_input(
        "Conductivity (μS/cm)", 
        value=default_values["Conductivity"], 
        format="%.4f",
        help="Batas: <= 400 μS/cm"
    )

with col3:
    input_data["Organic_carbon"] = st.number_input(
        "Organic Carbon (mg/L)", 
        value=default_values["Organic_carbon"], 
        format="%.4f",
        help="Batas: <= 2.0 mg/L"
    )
    input_data["Trihalomethanes"] = st.number_input(
        "Trihalomethanes (μg/L)", 
        value=default_values["Trihalomethanes"], 
        format="%.4f",
        help="Batas: <= 80 μg/L"
    )
    input_data["Turbidity"] = st.number_input(
        "Turbidity (NTU)", 
        value=default_values["Turbidity"], 
        format="%.4f",
        help="Batas: <= 5.0 NTU"
    )

# ============================================================
# TOMBOL PREDIKSI
# ============================================================
st.markdown("---")
col_btn, col_empty = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔍 Prediksi Kelayakan", type="primary", use_container_width=True)

# ============================================================
# HASIL PREDIKSI
# ============================================================
if predict_btn:
    with st.spinner("⏳ Memproses..."):
        result = predict_and_diagnose(input_data)

    st.markdown("---")
    st.subheader("📋 Hasil Prediksi")

    # ======== STATUS UTAMA ========
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if result["prediksi"] == "LAYAK":
            st.success("### ✅ LAYAK")
            st.caption("Air aman dikonsumsi")
        else:
            st.error("### ❌ TIDAK LAYAK")
            st.caption("Air tidak aman dikonsumsi")

    with col2:
        st.metric(
            "Probabilitas Layak",
            f"{result['probabilitas_layak']:.2%}"
        )

    with col3:
        st.metric(
            "Probabilitas Tidak Layak",
            f"{result['probabilitas_tidak_layak']:.2%}"
        )

    # ======== GAUGE ========
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['probabilitas_layak'] * 100,
        title={"text": "Probabilitas Layak (%)"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2E86C1"},
            "steps": [
                {"range": [0, 40], "color": "#F1948A"},
                {"range": [40, 60], "color": "#F9E79F"},
                {"range": [60, 100], "color": "#82E0AA"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ======== ANOMALI ========
    if result["anomali"]:
        st.warning(f"⚠️ Terdapat {len(result['anomali'])} parameter yang melanggar baku mutu!")

        df_anomali = pd.DataFrame(result["anomali"])
        df_anomali = df_anomali.rename(columns={
            "parameter": "Parameter",
            "nilai": "Nilai",
            "batas": "Batas Baku Mutu",
            "status": "Status"
        })
        st.dataframe(df_anomali, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Semua parameter dalam batas baku mutu!")

    # ======== REKOMENDASI SINGKAT ========
    if result["prediksi"] == "TIDAK LAYAK":
        st.info("💡 **Saran:** Lakukan pengolahan air (filtrasi, reverse osmosis, atau desinfeksi) sebelum dikonsumsi.")
    else:
        st.info("💡 **Saran:** Air layak dikonsumsi. Pastikan penyimpanan air tetap bersih.")

    # ======== DISCLAIMER ========
    st.caption(
        "⚠️ **Disclaimer**: Hasil ini bersifat indikatif dan tidak menggantikan pengujian laboratorium resmi."
    )

else:
    # ======== TAMPILAN AWAL ========
    st.info("👆 Masukkan parameter air di atas, lalu klik **'Prediksi Kelayakan'**")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("💧 Sistem Klasifikasi Kelayakan Air Minum • Machine Learning + Rule-Based Diagnostic System")
