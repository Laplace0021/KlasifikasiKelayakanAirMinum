import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Kelayakan Air Minum",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
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
# FUNGSI RBDS (Rule-Based Diagnostic System)
# ============================================================
def rbds_diagnose(sample: dict) -> list:
    """
    Rule-Based Diagnostic System (RBDS) Post-Filter
    Memeriksa parameter yang melanggar baku mutu WHO/EPA
    """
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
                "status": "DI BAWAH BATAS",
                "rekomendasi": f"Tingkatkan {label} hingga minimal {rules['min']} {unit}"
            })
        if rules["max"] is not None and val > rules["max"]:
            anomalies.append({
                "parameter": label,
                "nilai": round(val, 4),
                "batas": f"≤ {rules['max']} {unit}".strip(),
                "status": "MELEBIHI BATAS",
                "rekomendasi": f"Turunkan {label} hingga maksimal {rules['max']} {unit}"
            })
    return anomalies

def predict_and_diagnose(sample: dict) -> dict:
    """
    Melakukan prediksi ML dan RBDS jika hasil = tidak layak
    """
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
st.title("💧 Sistem Klasifikasi Kelayakan Air Minum")
st.markdown("""
<div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; margin-bottom:1.5rem;">
    <p style="margin:0; font-size:1.1rem;">
        Aplikasi ini mengimplementasikan <b>Machine Learning</b> dan 
        <b>Rule-Based Diagnostic System (RBDS)</b> untuk mengklasifikasikan 
        kelayakan air minum berdasarkan parameter fisikokimia.
    </p>
    <p style="margin:0.5rem 0 0 0; font-size:0.9rem; color:#555;">
        🤖 Model terbaik: <b>{}</b> • 📊 Mengacu pada standar WHO/EPA
    </p>
</div>
""".format(best_model_name), unsafe_allow_html=True)

# ============================================================
# METODOLOGI RINGKAS (Collapsible)
# ============================================================
with st.expander("📖 Tentang Sistem Ini"):
    st.markdown("""
    ### Metodologi Penelitian

    **1. Data & Preprocessing**
    - Dataset: Water Potability Dataset (3.276 sampel, 9 parameter)
    - Penanganan missing value: Median Imputation
    - Penanganan outlier: IQR Capping (Winsorization)
    - Pembagian data: 80% training, 20% testing (stratified)

    **2. Model Machine Learning**
    - Decision Tree (interpretabilitas tinggi)
    - Random Forest (ensemble learning)
    - XGBoost (gradient boosting)

    **3. Optimasi Hyperparameter**
    - GridSearchCV dengan 5-Fold Cross Validation
    - Metrik evaluasi: F1-Score (menangani class imbalance)

    **4. Rule-Based Diagnostic System (RBDS)**
    - Arsitektur Post-Filter (tidak mengubah prediksi ML)
    - IF-THEN rules berdasarkan standar WHO dan EPA
    - Memberikan diagnosis parameter yang melanggar baku mutu

    **5. Implementasi**
    - Framework: Streamlit
    - Model terbaik disimpan dalam format .pkl
    - Mendukung single prediction dan batch prediction
    """)

# ============================================================
# SIDEBAR - INPUT PARAMETER
# ============================================================
st.sidebar.header("📊 Input Parameter Air")
st.sidebar.markdown("Masukkan nilai 9 parameter fisikokimia air:")

# Inisialisasi input data
input_data = {}

# Nilai default berdasarkan dataset
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

# Informasi batas untuk tooltip
info_batas = {
    "ph": "Batas WHO: 6.5 – 8.5",
    "Hardness": "Batas: ≤ 300 mg/L",
    "Solids": "Batas: ≤ 500 mg/L (TDS)",
    "Chloramines": "Batas: ≤ 4.0 mg/L",
    "Sulfate": "Batas: ≤ 250 mg/L",
    "Conductivity": "Batas: ≤ 400 μS/cm",
    "Organic_carbon": "Batas: ≤ 2.0 mg/L",
    "Trihalomethanes": "Batas: ≤ 80 μg/L",
    "Turbidity": "Batas: ≤ 5.0 NTU"
}

col1, col2 = st.sidebar.columns(2)

for i, feat in enumerate(features):
    label = thresholds[feat]["label"]
    unit = thresholds[feat]["unit"]
    
    # Tampilkan batas sebagai tooltip
    help_text = info_batas.get(feat, "")
    
    if i % 2 == 0:
        val = col1.number_input(
            f"{label} ({unit})",
            value=default_values.get(feat, 0.0),
            format="%.4f",
            help=help_text
        )
    else:
        val = col2.number_input(
            f"{label} ({unit})",
            value=default_values.get(feat, 0.0),
            format="%.4f",
            help=help_text
        )
    input_data[feat] = val

# ============================================================
# TOMBOL PREDIKSI
# ============================================================
predict_btn = st.sidebar.button(
    "🔍 Prediksi & Diagnosa", 
    type="primary", 
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.caption(f"🤖 Model: **{best_model_name}**")
st.sidebar.caption("📋 Baku mutu: WHO & EPA")
st.sidebar.caption(f"📅 {datetime.now().strftime('%d %B %Y')}")

# ============================================================
# AREA HASIL PREDIKSI
# ============================================================
if predict_btn:
    with st.spinner("⏳ Memproses data..."):
        result = predict_and_diagnose(input_data)

    # ======== METRIK UTAMA ========
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if result["prediksi"] == "LAYAK":
            st.metric(
                "Status Kelayakan", 
                "✅ LAYAK", 
                delta="Aman dikonsumsi",
                delta_color="normal"
            )
        else:
            st.metric(
                "Status Kelayakan", 
                "❌ TIDAK LAYAK", 
                delta="Tidak aman",
                delta_color="inverse"
            )

    with col2:
        st.metric(
            "Probabilitas Layak",
            f"{result['probabilitas_layak']:.2%}",
            help="Probabilitas bahwa air layak menurut model ML"
        )

    with col3:
        st.metric(
            "Probabilitas Tidak Layak",
            f"{result['probabilitas_tidak_layak']:.2%}",
            help="Probabilitas bahwa air tidak layak menurut model ML"
        )

    with col4:
        if result["anomali"]:
            st.metric(
                "Parameter Bermasalah",
                f"{len(result['anomali'])}",
                delta="⚠️ Perlu perhatian",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Parameter Bermasalah",
                "0",
                delta="✅ Semua normal",
                delta_color="normal"
            )

    # ======== GAUGE CHART ========
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
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
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        if result["prediksi"] == "LAYAK":
            st.success("""
            ### ✅ Air LAYAK Dikonsumsi
            
            Model Machine Learning memprediksi bahwa air ini **LAYAK** 
            untuk dikonsumsi berdasarkan parameter fisikokimia yang diinput.
            
            **Probabilitas layak:** {:.2%}
            """.format(result['probabilitas_layak']))
        else:
            st.error("""
            ### ❌ Air TIDAK LAYAK Dikonsumsi
            
            Model Machine Learning memprediksi bahwa air ini **TIDAK LAYAK** 
            untuk dikonsumsi berdasarkan parameter fisikokimia yang diinput.
            
            **Probabilitas tidak layak:** {:.2%}
            """.format(result['probabilitas_tidak_layak']))

    # ======== PARAMETER INPUT ========
    with st.expander("📋 Detail Parameter Input", expanded=True):
        df_input = pd.DataFrame({
            "Parameter": [thresholds[f]["label"] for f in features],
            "Nilai": [input_data[f] for f in features],
            "Satuan": [thresholds[f]["unit"] for f in features],
            "Batas Minimum": [thresholds[f]["min"] if thresholds[f]["min"] is not None else "-" for f in features],
            "Batas Maksimum": [thresholds[f]["max"] if thresholds[f]["max"] is not None else "-" for f in features]
        })
        
        # Warna berdasarkan status
        def color_cell(val, param_name, sample):
            rules = thresholds[param_name]
            if rules["min"] is not None and val < rules["min"]:
                return "background-color: #F1948A"
            if rules["max"] is not None and val > rules["max"]:
                return "background-color: #F1948A"
            return ""
        
        st.dataframe(df_input, use_container_width=True, hide_index=True)

    # ======== ANOMALI RBDS ========
    if result["anomali"]:
        st.subheader("⚠️ Anomali Parameter Terdeteksi")
        st.warning(f"Terdapat **{len(result['anomali'])}** parameter yang melanggar baku mutu!")

        df_anomali = pd.DataFrame(result["anomali"])
        df_anomali = df_anomali.rename(columns={
            "parameter": "Parameter",
            "nilai": "Nilai",
            "batas": "Batas Baku Mutu",
            "status": "Status",
            "rekomendasi": "Rekomendasi"
        })
        st.dataframe(df_anomali, use_container_width=True, hide_index=True)

        # Visualisasi anomali
        fig_bar = go.Figure()
        for a in result["anomali"]:
            status_color = "#E74C3C" if a["status"] == "MELEBIHI BATAS" else "#F39C12"
            fig_bar.add_trace(go.Bar(
                x=[a["parameter"]],
                y=[a["nilai"]],
                name=a["parameter"],
                text=[f"{a['nilai']} ({a['status']})"],
                textposition="outside",
                marker_color=status_color,
                hovertemplate=f"{a['parameter']}<br>Nilai: {a['nilai']}<br>Batas: {a['batas']}<br>Status: {a['status']}<extra></extra>"
            ))

        fig_bar.update_layout(
            title="Nilai Parameter yang Bermasalah",
            yaxis_title="Nilai",
            height=350,
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.success("✅ Semua parameter dalam batas baku mutu air minum!")

    # ======== REKOMENDASI ========
    with st.expander("💡 Rekomendasi & Tindakan", expanded=result["prediksi"] == "TIDAK LAYAK"):
        if result["prediksi"] == "LAYAK":
            st.info("""
            ### ✅ Air LAYAK Dikonsumsi
            
            **Rekomendasi:**
            - ✅ Air dapat langsung dikonsumsi atau digunakan untuk kebutuhan rumah tangga
            - ✅ Pastikan penyimpanan air bersih dan terlindung dari kontaminasi
            - ✅ Lakukan pengujian berkala untuk memastikan kualitas tetap terjaga
            - ✅ Gunakan sistem filtrasi sederhana untuk menjaga kualitas
            """)
        else:
            st.error("""
            ### ❌ Air TIDAK LAYAK Dikonsumsi
            
            **Rekomendasi Penanganan:**
            
            1. **Jangan mengonsumsi air ini secara langsung**
            2. **Lakukan pengolahan air:**
               - **Filtrasi** untuk menghilangkan partikel tersuspensi
               - **Reverse Osmosis** untuk menurunkan kadar garam terlarut (TDS)
               - **Desinfeksi** (UV/Klorinasi) untuk membunuh mikroorganisme
            3. **Lakukan pengujian ulang** setelah pengolahan
            4. **Konsultasikan** dengan otoritas kesehatan setempat
            """)
            
            if result["anomali"]:
                st.write("**Prioritas Penanganan Parameter:**")
                for i, a in enumerate(result["anomali"], 1):
                    st.write(f"{i}. **{a['parameter']}**: {a['nilai']} (batas: {a['batas']}) → {a['rekomendasi']}")

    # ======== DISCLAIMER ========
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer**: Aplikasi ini bersifat demonstratif untuk tujuan edukasi dan penelitian. 
    Hasil prediksi dan diagnosa **tidak menggantikan** pengujian laboratorium resmi. 
    Selalu konsultasikan dengan otoritas kesehatan setempat untuk penentuan kelayakan air minum yang akurat.
    """)

else:
    # ======== TAMPILAN AWAL ========
    st.info("👈 Masukkan nilai parameter air di sidebar, lalu klik **'Prediksi & Diagnosa'**")
    
    # Tampilkan diagram alir metodologi
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Metodologi Penelitian
        
