import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Batch Prediction - Klasifikasi Air Minum",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Batch Prediction")
st.markdown("Unggah file CSV untuk melakukan prediksi kelayakan air minum secara batch.")

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
                "batas": f"≥ {rules['min']} {unit}".strip(),
                "status": "DI BAWAH BATAS"
            })
        if rules["max"] is not None and val > rules["max"]:
            anomalies.append({
                "parameter": label,
                "nilai": round(val, 4),
                "batas": f"≤ {rules['max']} {unit}".strip(),
                "status": "MELEBIHI BATAS"
            })
    return anomalies

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan prediksi batch dan RBDS untuk setiap sampel
    """
    # Prediksi ML
    X = df[features]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Buat hasil
    results = df.copy()
    results["Prediksi_Label"] = predictions
    results["Prediksi"] = predictions.map({1: "LAYAK", 0: "TIDAK LAYAK"})
    results["Prob_Layak"] = probabilities[:, 1]
    results["Prob_Tidak_Layak"] = probabilities[:, 0]
    
    # RBDS untuk setiap sampel yang tidak layak
    results["Jumlah_Anomali"] = 0
    results["Detail_Anomali"] = ""
    
    for idx, row in results.iterrows():
        if row["Prediksi_Label"] == 0:
            sample = row[features].to_dict()
            anomalies = rbds_diagnose(sample)
            results.at[idx, "Jumlah_Anomali"] = len(anomalies)
            if anomalies:
                detail = "; ".join([f"{a['parameter']}: {a['nilai']} ({a['status']})" for a in anomalies])
                results.at[idx, "Detail_Anomali"] = detail
    
    return results

# ============================================================
# UPLOAD FILE
# ============================================================
uploaded_file = st.file_uploader(
    "Unggah file CSV",
    type=["csv"],
    help="File harus memiliki kolom: " + ", ".join(features)
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validasi kolom
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Kolom tidak ditemukan: {missing_cols}")
            st.stop()
        
        st.success(f"✅ File berhasil diunggah! {len(df)} sampel ditemukan.")
        
        # Tampilkan preview
        with st.expander("📋 Preview Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # ============================================================
        # PROSES PREDIKSI
        # ============================================================
        if st.button("🚀 Proses Prediksi Batch", type="primary"):
            with st.spinner("⏳ Memproses data..."):
                results = predict_batch(df)
            
            # ======== METRIK ========
            col1, col2, col3, col4 = st.columns(4)
            
            layak_count = results[results["Prediksi_Label"] == 1].shape[0]
            tidak_layak_count = results[results["Prediksi_Label"] == 0].shape[0]
            anomali_count = results[results["Jumlah_Anomali"] > 0].shape[0]
            
            with col1:
                st.metric("Total Sampel", len(results))
            with col2:
                st.metric("✅ Layak", layak_count, delta=f"{layak_count/len(results):.1%}")
            with col3:
                st.metric("❌ Tidak Layak", tidak_layak_count, delta=f"{tidak_layak_count/len(results):.1%}", delta_color="inverse")
            with col4:
                st.metric("⚠️ Bermasalah", anomali_count, delta=f"{anomali_count/len(results):.1%}", delta_color="inverse")
            
            # ======== VISUALISASI ========
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribusi Prediksi", "Distribusi Probabilitas Layak"),
                specs=[[{"type": "pie"}, {"type": "histogram"}]]
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(
                    labels=["LAYAK", "TIDAK LAYAK"],
                    values=[layak_count, tidak_layak_count],
                    marker_colors=["#82E0AA", "#F1948A"],
                    textinfo="label+percent"
                ),
                row=1, col=1
            )
            
            # Histogram probabilitas
            fig.add_trace(
                go.Histogram(
                    x=results["Prob_Layak"],
                    nbinsx=20,
                    marker_color="#2E86C1",
                    name="Probabilitas Layak"
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # ======== TABEL HASIL ========
            st.subheader("📊 Hasil Prediksi")
            
            # Pilih kolom untuk ditampilkan
            display_cols = features + ["Prediksi", "Prob_Layak", "Jumlah_Anomali", "Detail_Anomali"]
            display_df = results[display_cols].copy()
            
            # Format kolom
            display_df["Prob_Layak"] = display_df["Prob_Layak"].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Prediksi": st.column_config.TextColumn("Status", width="small"),
                    "Prob_Layak": st.column_config.TextColumn("Prob. Layak", width="small"),
                    "Jumlah_Anomali": st.column_config.NumberColumn("Anomali", width="small"),
                    "Detail_Anomali": st.column_config.TextColumn("Detail Anomali", width="large")
                }
            )
            
            # ======== DOWNLOAD ========
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name="hasil_prediksi_air_minum.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # ======== STATISTIK ANOMALI ========
            if anomali_count > 0:
                st.subheader("⚠️ Ringkasan Anomali Parameter")
                
                # Hitung frekuensi anomali per parameter
                anomaly_params = []
                for detail in results[results["Jumlah_Anomali"] > 0]["Detail_Anomali"]:
                    for part in detail.split("; "):
                        if ": " in part:
                            param = part.split(":")[0]
                            anomaly_params.append(param)
                
                if anomaly_params:
                    param_counts = pd.Series(anomaly_params).value_counts().reset_index()
                    param_counts.columns = ["Parameter", "Frekuensi"]
                    
                    fig_bar = go.Figure(go.Bar(
                        x=param_counts["Parameter"],
                        y=param_counts["Frekuensi"],
                        marker_color="#E74C3C",
                        text=param_counts["Frekuensi"],
                        textposition="outside"
                    ))
                    fig_bar.update_layout(
                        title="Parameter yang Sering Melanggar Baku Mutu",
                        yaxis_title="Frekuensi Pelanggaran",
                        height=350,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")

else:
    st.info("📂 Silakan unggah file CSV untuk memulai prediksi batch.")
    
    # Tampilkan format contoh
    with st.expander("📄 Format File CSV yang Diperlukan"):
        st.markdown("""
        File CSV harus memiliki kolom berikut (dengan header yang sama persis):
        
