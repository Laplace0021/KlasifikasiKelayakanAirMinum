import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

try:
    model = load_model()
    config = load_config()
    features = config["fitur"]
    thresholds = config["thresholds_rbds"]
    best_model_name = config.get("nama_model", "Decision Tree")
except Exception as e:
    st.error(f"❌ Gagal memuat model atau konfigurasi: {str(e)}")
    st.stop()

# ============================================================
# FUNGSI RBDS (Rule-Based Diagnostic System)
# ============================================================
def rbds_diagnose(sample: dict) -> list:
    """
    Rule-Based Diagnostic System (RBDS)
    Memeriksa parameter yang melanggar baku mutu WHO/EPA
    """
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

def predict_and_diagnose(sample: dict) -> dict:
    """
    Melakukan prediksi ML dan RBDS jika hasil = tidak layak
    """
    try:
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
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {str(e)}")
        return None

# ============================================================
# HEADER
# ============================================================
st.title("💧 Klasifikasi Kelayakan Air Minum")
st.markdown(f"""
<div style="background-color:#f0f8ff; padding:1rem; border-radius:10px; margin-bottom:1.5rem; border-left:5px solid #2E86C1;">
    <p style="margin:0; font-size:1rem;">
        🤖 <b>Model terbaik:</b> {best_model_name} &nbsp;|&nbsp; 
        📊 <b>Mengacu pada standar:</b> WHO & EPA
    </p>
    <p style="margin:0.3rem 0 0 0; font-size:0.9rem; color:#555;">
        Masukkan 9 parameter fisikokimia air untuk mengetahui kelayakan konsumsi
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT PARAMETER - 3 Kolom
# ============================================================
st.subheader("📊 Masukkan Parameter Air")

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
    "Organic_carbon": "Batas: ≤ 2.0 mg/L (Karbon Organik Total)",
    "Trihalomethanes": "Batas: ≤ 80 μg/L",
    "Turbidity": "Batas: ≤ 5.0 NTU"
}

input_data = {}

col1, col2, col3 = st.columns(3)

with col1:
    input_data["ph"] = st.number_input(
        "pH", 
        value=default_values["ph"], 
        format="%.4f",
        help=info_batas["ph"],
        step=0.1
    )
    input_data["Hardness"] = st.number_input(
        "Hardness (mg/L)", 
        value=default_values["Hardness"], 
        format="%.4f",
        help=info_batas["Hardness"],
        step=1.0
    )
    input_data["Solids"] = st.number_input(
        "Solids / TDS (mg/L)", 
        value=default_values["Solids"], 
        format="%.4f",
        help=info_batas["Solids"],
        step=100.0
    )

with col2:
    input_data["Chloramines"] = st.number_input(
        "Chloramines (mg/L)", 
        value=default_values["Chloramines"], 
        format="%.4f",
        help=info_batas["Chloramines"],
        step=0.1
    )
    input_data["Sulfate"] = st.number_input(
        "Sulfate (mg/L)", 
        value=default_values["Sulfate"], 
        format="%.4f",
        help=info_batas["Sulfate"],
        step=1.0
    )
    input_data["Conductivity"] = st.number_input(
        "Conductivity (μS/cm)", 
        value=default_values["Conductivity"], 
        format="%.4f",
        help=info_batas["Conductivity"],
        step=1.0
    )

with col3:
    input_data["Organic_carbon"] = st.number_input(
        "Organic Carbon (mg/L)", 
        value=default_values["Organic_carbon"], 
        format="%.4f",
        help=info_batas["Organic_carbon"],
        step=0.1
    )
    input_data["Trihalomethanes"] = st.number_input(
        "Trihalomethanes (μg/L)", 
        value=default_values["Trihalomethanes"], 
        format="%.4f",
        help=info_batas["Trihalomethanes"],
        step=1.0
    )
    input_data["Turbidity"] = st.number_input(
        "Turbidity (NTU)", 
        value=default_values["Turbidity"], 
        format="%.4f",
        help=info_batas["Turbidity"],
        step=0.1
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
    with st.spinner("⏳ Memproses data..."):
        result = predict_and_diagnose(input_data)

    if result is None:
        st.stop()

    st.markdown("---")
    st.subheader("📋 Hasil Prediksi")

    # ======== STATUS UTAMA ========
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if result["prediksi"] == "LAYAK":
            st.success("### ✅ LAYAK")
            st.caption("Air aman untuk dikonsumsi")
        else:
            st.error("### ❌ TIDAK LAYAK")
            st.caption("Air tidak aman untuk dikonsumsi")

    with col2:
        st.metric(
            "Probabilitas Layak",
            f"{result['probabilitas_layak']:.2%}",
            help="Semakin tinggi, semakin yakin model bahwa air layak"
        )

    with col3:
        st.metric(
            "Probabilitas Tidak Layak",
            f"{result['probabilitas_tidak_layak']:.2%}",
            help="Semakin tinggi, semakin yakin model bahwa air tidak layak"
        )

    # ======== PROGRESS BAR (Visualisasi Probabilitas) ========
    st.subheader("📊 Visualisasi Probabilitas")
    
    prob_layak = result['probabilitas_layak']
    prob_tidak = result['probabilitas_tidak_layak']
    
    # Warna berdasarkan probabilitas
    if prob_layak >= 0.6:
        color = "#28a745"  # Hijau
        status_text = "🟢 Kemungkinan LAYAK"
    elif prob_layak >= 0.4:
        color = "#ffc107"  # Kuning
        status_text = "🟡 Kemungkinan Ragu-ragu"
    else:
        color = "#dc3545"  # Merah
        status_text = "🔴 Kemungkinan TIDAK LAYAK"
    
    # Progress bar custom dengan HTML/CSS
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">Probabilitas Layak</span>
            <span style="font-weight: bold; color: {color};">{prob_layak:.1%}</span>
        </div>
        <div style="background-color: #e9ecef; border-radius: 10px; height: 30px; overflow: hidden; position: relative;">
            <div style="background: linear-gradient(90deg, #dc3545, #ffc107, #28a745); 
                        width: 100%; height: 100%; border-radius: 10px; opacity: 0.3;">
            </div>
            <div style="background-color: {color}; 
                        width: {prob_layak*100}%; height: 100%; 
                        border-radius: 10px; 
                        position: absolute; top: 0; left: 0;
                        transition: width 0.5s ease;">
            </div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        font-weight: bold; font-size: 14px; color: #333;">
                {status_text}
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.8rem; color: #666;">
            <span>0% (Tidak Layak)</span>
            <span>50%</span>
            <span>100% (Layak)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======== TABEL PARAMETER INPUT ========
    with st.expander("📋 Detail Parameter Input", expanded=False):
        df_input = pd.DataFrame({
            "Parameter": [thresholds[f]["label"] for f in features],
            "Nilai": [input_data[f] for f in features],
            "Satuan": [thresholds[f]["unit"] for f in features],
            "Batas Minimum": [thresholds[f]["min"] if thresholds[f]["min"] is not None else "-" for f in features],
            "Batas Maksimum": [thresholds[f]["max"] if thresholds[f]["max"] is not None else "-" for f in features]
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)

    # ======== ANOMALI RBDS ========
    if result["anomali"]:
        st.warning(f"⚠️ Terdapat **{len(result['anomali'])}** parameter yang melanggar baku mutu!")

        df_anomali = pd.DataFrame(result["anomali"])
        df_anomali = df_anomali.rename(columns={
            "parameter": "Parameter",
            "nilai": "Nilai",
            "batas": "Batas Baku Mutu",
            "status": "Status"
        })
        
        # Tambahkan kolom status dengan warna
        def color_status(status):
            if status == "MELEBIHI BATAS":
                return "🔴 MELEBIHI BATAS"
            else:
                return "🟡 DI BAWAH BATAS"
        
        df_anomali["Status"] = df_anomali["Status"].apply(color_status)
        st.dataframe(df_anomali, use_container_width=True, hide_index=True)
        
        # Tampilkan rekomendasi per parameter
        st.markdown("**📝 Rekomendasi Penanganan:**")
        for i, row in df_anomali.iterrows():
            if "MELEBIHI BATAS" in row["Status"]:
                st.write(f"- **{row['Parameter']}**: Nilai {row['Nilai']} melebihi batas maksimum {row['Batas Baku Mutu']}. Perlu dilakukan pengolahan untuk menurunkan kadar {row['Parameter']}.")
            else:
                st.write(f"- **{row['Parameter']}**: Nilai {row['Nilai']} di bawah batas minimum {row['Batas Baku Mutu']}. Perlu dilakukan pengolahan untuk menaikkan kadar {row['Parameter']}.")
    else:
        st.success("✅ Semua parameter dalam batas baku mutu air minum!")

    # ======== REKOMENDASI UMUM ========
    with st.expander("💡 Rekomendasi & Informasi", expanded=False):
        if result["prediksi"] == "LAYAK":
            st.info("""
            ### ✅ Air LAYAK Dikonsumsi
            
            **Rekomendasi:**
            - ✅ Air dapat langsung dikonsumsi atau digunakan untuk kebutuhan rumah tangga
            - ✅ Pastikan penyimpanan air bersih dan terlindung dari kontaminasi
            - ✅ Lakukan pengujian berkala untuk memastikan kualitas tetap terjaga
            """)
        else:
            st.error("""
            ### ❌ Air TIDAK LAYAK Dikonsumsi
            
            **Rekomendasi Penanganan:**
            
            1. **Jangan mengonsumsi air ini secara langsung**
            2. **Lakukan pengolahan air:**
               - **Filtrasi** untuk menghilangkan partikel tersuspensi
               - **Reverse Osmosis** untuk menurunkan kadar garam terlarut
               - **Desinfeksi** (UV/Klorinasi) untuk membunuh mikroorganisme
            3. **Lakukan pengujian ulang** setelah pengolahan
            4. **Konsultasikan** dengan otoritas kesehatan setempat
            """)

    # ======== DISCLAIMER ========
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: Hasil prediksi bersifat indikatif dan "
        "**tidak menggantikan** pengujian laboratorium resmi. "
        "Konsultasikan dengan otoritas kesehatan untuk kepastian kelayakan air minum."
    )

else:
    # ======== TAMPILAN AWAL ========
    st.info("👆 Masukkan parameter air di atas, lalu klik **'Prediksi Kelayakan'**")
    
    # Tampilkan informasi singkat
    with st.expander("📖 Tentang Sistem Ini", expanded=False):
        st.markdown("""
        ### 💧 Sistem Klasifikasi Kelayakan Air Minum
        
        Aplikasi ini menggunakan **Machine Learning** dan **Rule-Based Diagnostic System (RBDS)** 
        untuk mengklasifikasikan kelayakan air minum.
        
        **Parameter yang digunakan:**
        - pH, Hardness, Solids (TDS), Chloramines, Sulfate
        - Conductivity, Organic Carbon, Trihalomethanes, Turbidity
        
        **Standar Baku Mutu:**
        Mengacu pada standar **WHO** (World Health Organization) dan **EPA** (Environmental Protection Agency).
        
        **Model Terbaik:** Decision Tree (F1-Score: 0.478)
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("💧 Sistem Klasifikasi Kelayakan Air Minum • Machine Learning + Rule-Based Diagnostic System")
