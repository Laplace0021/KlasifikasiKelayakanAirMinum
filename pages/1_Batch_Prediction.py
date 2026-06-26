import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Batch Prediction")
st.caption("Unggah file CSV untuk prediksi massal kelayakan air minum")

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

try:
    model = load_model()
    config = load_config()
    features = config["fitur"]
    thresholds = config["thresholds_rbds"]
except Exception as e:
    st.error(f"❌ Gagal memuat model: {str(e)}")
    st.stop()

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

        if rules.get("min") is not None and val < rules["min"]:
            anomalies.append({
                "parameter": label,
                "nilai": round(val, 4),
                "batas": f">= {rules['min']} {unit}".strip(),
                "status": "DI BAWAH BATAS"
            })
        if rules.get("max") is not None and val > rules["max"]:
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
uploaded_file = st.file_uploader(
    "Unggah file CSV",
    type=["csv"],
    help="File harus memiliki kolom: " + ", ".join(features)
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Kolom tidak ditemukan: {missing_cols}")
            st.stop()
        
        st.success(f"✅ {len(df)} sampel ditemukan")
        
        with st.expander("📋 Preview Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🚀 Proses Prediksi", type="primary", use_container_width=True):
            with st.spinner("⏳ Memproses data..."):
                results = predict_batch(df)
            
            # ======== METRIK ========
            col1, col2, col3, col4 = st.columns(4)
            layak = results[results["Prediksi"] == "LAYAK"].shape[0]
            tidak = results[results["Prediksi"] == "TIDAK LAYAK"].shape[0]
            anomali = results[results["Jumlah_Anomali"] > 0].shape[0]
            
            with col1:
                st.metric("Total Sampel", len(results))
            with col2:
                st.metric("✅ Layak", layak, delta=f"{layak/len(results):.1%}")
            with col3:
                st.metric("❌ Tidak Layak", tidak, delta=f"{tidak/len(results):.1%}", delta_color="inverse")
            with col4:
                st.metric("⚠️ Bermasalah", anomali, delta=f"{anomali/len(results):.1%}", delta_color="inverse")
            
            # ======== TABEL HASIL ========
            st.subheader("📊 Hasil Prediksi")
            
            display_cols = features + ["Prediksi", "Prob_Layak", "Jumlah_Anomali", "Detail_Anomali"]
            display_df = results[display_cols].copy()
            display_df["Prob_Layak"] = display_df["Prob_Layak"].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # ======== DOWNLOAD ========
            csv = results.to_csv(index=False)
            st.download_button(
                "📥 Download Hasil (CSV)",
                data=csv,
                file_name="hasil_prediksi_air_minum.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # ======== STATISTIK ANOMALI ========
            if anomali > 0:
                st.subheader("⚠️ Ringkasan Anomali Parameter")
                
                anomaly_params = []
                for detail in results[results["Jumlah_Anomali"] > 0]["Detail_Anomali"]:
                    for part in str(detail).split("; "):
                        if ": " in part:
                            param = part.split(":")[0]
                            anomaly_params.append(param)
                
                if anomaly_params:
                    param_counts = pd.Series(anomaly_params).value_counts().reset_index()
                    param_counts.columns = ["Parameter", "Frekuensi"]
                    st.dataframe(param_counts, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")

else:
    st.info("📂 Upload file CSV untuk memulai prediksi batch")
    
    with st.expander("📄 Format File CSV", expanded=False):
        st.markdown("""
        File CSV harus memiliki kolom berikut:
        ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
""")

sample_data = {
    "ph": [7.0, 8.5, 6.2],
    "Hardness": [200.0, 180.0, 250.0],
    "Solids": [20000.0, 15000.0, 30000.0],
    "Chloramines": [7.0, 6.5, 8.0],
    "Sulfate": [330.0, 300.0, 350.0],
    "Conductivity": [400.0, 350.0, 450.0],
    "Organic_carbon": [14.0, 12.0, 18.0],
    "Trihalomethanes": [66.0, 60.0, 80.0],
    "Turbidity": [4.0, 3.5, 5.5]
}
df_sample = pd.DataFrame(sample_data)
csv_sample = df_sample.to_csv(index=False)
st.download_button(
    "📥 Download Contoh CSV",
    data=csv_sample,
    file_name="contoh_data_air_minum.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("📊 Batch Prediction • Sistem Klasifikasi Kelayakan Air Minum")
