import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaCheck — Klasifikasi Kualitas Air",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
    }

    /* Main area */
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #075985 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .hero-banner h1 {
        color: #fff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.2;
    }
    .hero-banner p { color: #bae6fd !important; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Result banners */
    .result-layak {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 5px solid #16a34a;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
    }
    .result-layak h2 { color: #15803d !important; margin: 0 !important; font-size: 1.5rem !important; }
    .result-layak p  { color: #166534 !important; margin: 0.3rem 0 0; font-size: 0.9rem; }

    .result-tidak {
        background: linear-gradient(135deg, #fef2f2, #fecaca);
        border-left: 5px solid #dc2626;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
    }
    .result-tidak h2 { color: #b91c1c !important; margin: 0 !important; font-size: 1.5rem !important; }
    .result-tidak p  { color: #7f1d1d !important; margin: 0.3rem 0 0; font-size: 0.9rem; }

    /* Anomaly tag */
    .anomaly-tag {
        display: inline-block;
        background: #fef9c3;
        border: 1px solid #fde047;
        border-radius: 6px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #713f12;
        margin: 0.2rem;
    }

    /* Metric row */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .metric-pill {
        background: #e0f2fe;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #0369a1;
    }

    /* Section label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }

    /* Divider */
    .divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.4rem 0; }

    /* Table styling */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Number input labels */
    label { font-size: 0.85rem !important; font-weight: 500 !important; color: #334155 !important; }

    /* Warning disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #78350f;
        margin-top: 1.5rem;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Konstanta ───────────────────────────────────────────────────────────────
FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

FEATURE_META = {
    'ph':              {'label': 'pH',              'unit': '',       'min': 0.0,   'max': 14.0,    'default': 7.0,    'step': 0.01},
    'Hardness':        {'label': 'Hardness',        'unit': 'mg/L',   'min': 0.0,   'max': 800.0,   'default': 196.0,  'step': 0.1},
    'Solids':          {'label': 'Solids (TDS)',     'unit': 'mg/L',   'min': 0.0,   'max': 60000.0, 'default': 20000.0,'step': 1.0},
    'Chloramines':     {'label': 'Chloramines',      'unit': 'mg/L',   'min': 0.0,   'max': 15.0,    'default': 7.0,    'step': 0.01},
    'Sulfate':         {'label': 'Sulfate',          'unit': 'mg/L',   'min': 0.0,   'max': 600.0,   'default': 333.0,  'step': 0.1},
    'Conductivity':    {'label': 'Conductivity',     'unit': 'μS/cm',  'min': 0.0,   'max': 800.0,   'default': 426.0,  'step': 0.1},
    'Organic_carbon':  {'label': 'Organic Carbon',   'unit': 'mg/L',   'min': 0.0,   'max': 30.0,    'default': 14.0,   'step': 0.01},
    'Trihalomethanes': {'label': 'Trihalomethanes',  'unit': 'μg/L',   'min': 0.0,   'max': 130.0,   'default': 66.0,   'step': 0.01},
    'Turbidity':       {'label': 'Turbidity',        'unit': 'NTU',    'min': 0.0,   'max': 7.0,     'default': 3.9,    'step': 0.01},
}

THRESHOLDS = {
    'ph':              {'min': 6.5,  'max': 8.5,   'unit': '',      'label': 'pH'},
    'Hardness':        {'min': None, 'max': 300.0,  'unit': 'mg/L',  'label': 'Hardness'},
    'Solids':          {'min': None, 'max': 500.0,  'unit': 'mg/L',  'label': 'Solids (TDS)'},
    'Chloramines':     {'min': None, 'max': 4.0,    'unit': 'mg/L',  'label': 'Chloramines'},
    'Sulfate':         {'min': None, 'max': 250.0,  'unit': 'mg/L',  'label': 'Sulfate'},
    'Conductivity':    {'min': None, 'max': 400.0,  'unit': 'μS/cm', 'label': 'Conductivity'},
    'Organic_carbon':  {'min': None, 'max': 2.0,    'unit': 'mg/L',  'label': 'Organic Carbon'},
    'Trihalomethanes': {'min': None, 'max': 80.0,   'unit': 'μg/L',  'label': 'Trihalomethanes'},
    'Turbidity':       {'min': None, 'max': 5.0,    'unit': 'NTU',   'label': 'Turbidity'},
}

# ─── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        with open('model_info.json') as f:
            info = json.load(f)
        return model, info
    except FileNotFoundError:
        return None, None

model, model_info = load_model()

# ─── RBDS ────────────────────────────────────────────────────────────────────
def rbds_diagnose(sample: dict) -> list:
    anomalies = []
    for param, rules in THRESHOLDS.items():
        val = sample.get(param)
        if val is None:
            continue
        label = rules['label']
        unit  = rules['unit']
        if rules['min'] is not None and val < rules['min']:
            anomalies.append({
                'Parameter': label,
                'Nilai': round(val, 4),
                'Batas Aman': f"≥ {rules['min']} {unit}".strip(),
                'Status': '⬇ DI BAWAH BATAS'
            })
        if rules['max'] is not None and val > rules['max']:
            anomalies.append({
                'Parameter': label,
                'Nilai': round(val, 4),
                'Batas Aman': f"≤ {rules['max']} {unit}".strip(),
                'Status': '⬆ MELEBIHI BATAS'
            })
    return anomalies


def predict_and_diagnose(sample: dict) -> dict:
    input_df = pd.DataFrame([sample])[FEATURES]
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    result = {
        'label': int(prediction),
        'prediksi': 'LAYAK' if prediction == 1 else 'TIDAK LAYAK',
        'prob_layak': round(float(probability[1]), 4),
        'prob_tidak': round(float(probability[0]), 4),
        'anomali': []
    }
    if prediction == 0:
        result['anomali'] = rbds_diagnose(sample)
    return result

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 AquaCheck")
    st.markdown("Sistem klasifikasi kelayakan air minum berbasis **Machine Learning + RBDS**.")
    st.markdown("---")

    mode = st.radio(
        "Mode Analisis",
        ["🔬 Prediksi Tunggal", "📂 Prediksi Batch (CSV)"],
        index=0
    )

    st.markdown("---")

    if model_info:
        st.markdown("**Model Aktif**")
        st.markdown(f"`{model_info.get('nama_model', 'Unknown')}`")
    else:
        st.warning("Model belum dimuat.\nPastikan `best_model.pkl` dan `model_info.json` ada di direktori yang sama.")

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#475569;'>"
        "Sistem ini berfungsi sebagai alat skrining awal. "
        "Bukan pengganti uji laboratorium bersertifikasi.</span>",
        unsafe_allow_html=True
    )

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div style="font-size:2.8rem;">💧</div>
    <div>
        <h1>AquaCheck</h1>
        <p>Klasifikasi kelayakan air minum berbasis Machine Learning &amp; Rule-Based Diagnostic System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Model tidak ditemukan ───────────────────────────────────────────────────
if model is None:
    st.error(
        "**Model tidak ditemukan.** "
        "Pastikan file `best_model.pkl` dan `model_info.json` berada di direktori yang sama dengan `app.py`. "
        "Jalankan notebook Colab terlebih dahulu untuk menghasilkan file tersebut."
    )
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 — PREDIKSI TUNGGAL
# ═════════════════════════════════════════════════════════════════════════════
if mode == "🔬 Prediksi Tunggal":

    st.markdown('<div class="section-label">Input Parameter Fisikokimia</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    inputs = {}

    param_list = list(FEATURE_META.items())
    for i, (key, meta) in enumerate(param_list):
        col = [col1, col2, col3][i % 3]
        label_str = f"{meta['label']} ({meta['unit']})" if meta['unit'] else meta['label']
        with col:
            inputs[key] = st.number_input(
                label_str,
                min_value=float(meta['min']),
                max_value=float(meta['max']),
                value=float(meta['default']),
                step=float(meta['step']),
                format="%.4f",
                key=key
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("🔍 Analisis Kualitas Air", use_container_width=True, type="primary"):
        result = predict_and_diagnose(inputs)

        # ── Hasil prediksi ──
        if result['label'] == 1:
            st.markdown(f"""
            <div class="result-layak">
                <h2>✅ Air LAYAK Dikonsumsi</h2>
                <p>Model memprediksi sampel ini memenuhi standar kelayakan air minum.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-tidak">
                <h2>⛔ Air TIDAK LAYAK Dikonsumsi</h2>
                <p>Model memprediksi sampel ini tidak memenuhi standar kelayakan air minum.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probabilitas ──
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Probabilitas Layak</div>
                <div class="card-value" style="color:#16a34a;">{result['prob_layak']:.2%}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Probabilitas Tidak Layak</div>
                <div class="card-value" style="color:#dc2626;">{result['prob_tidak']:.2%}</div>
            </div>""", unsafe_allow_html=True)

        # ── RBDS Diagnostik ──
        if result['label'] == 0:
            st.markdown('<div class="section-label" style="margin-top:1.2rem;">Diagnostik RBDS — Parameter Anomali</div>', unsafe_allow_html=True)

            if result['anomali']:
                st.markdown(
                    "Parameter berikut melanggar ambang batas standar baku mutu WHO/EPA:",
                    unsafe_allow_html=True
                )
                df_anomali = pd.DataFrame(result['anomali'])
                st.dataframe(df_anomali, use_container_width=True, hide_index=True)

                st.markdown('<div class="section-label" style="margin-top:0.8rem;">Ringkasan</div>', unsafe_allow_html=True)
                tags_html = "".join(
                    f'<span class="anomaly-tag">⚠ {a["Parameter"]}</span>'
                    for a in result['anomali']
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info(
                    "Meskipun model memprediksi air tidak layak, tidak ada parameter "
                    "yang secara individual melanggar ambang batas baku mutu. "
                    "Kemungkinan disebabkan oleh kombinasi parameter yang kompleks."
                )

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Perhatian:</strong> Hasil prediksi ini merupakan skrining awal dan bukan pengganti 
            analisis laboratorium bersertifikasi. Selalu lakukan verifikasi lebih lanjut sebelum mengambil 
            keputusan terkait konsumsi air.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 — PREDIKSI BATCH
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-label">Upload File CSV</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload file `.csv` dengan kolom: "
        "`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`"
    )

    # Template download
    template_df = pd.DataFrame([{k: FEATURE_META[k]['default'] for k in FEATURES}])
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇ Download Template CSV",
        data=csv_template,
        file_name="template_input.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)

            # Validasi kolom
            missing_cols = [c for c in FEATURES if c not in df_input.columns]
            if missing_cols:
                st.error(f"Kolom tidak ditemukan: {missing_cols}")
                st.stop()

            st.success(f"✅ File berhasil dimuat — **{len(df_input)} sampel** terdeteksi.")
            st.markdown('<div class="section-label">Preview Data</div>', unsafe_allow_html=True)
            st.dataframe(df_input[FEATURES].head(5), use_container_width=True, hide_index=True)

            if st.button("🔍 Proses Semua Sampel", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    results = []
                    for _, row in df_input[FEATURES].iterrows():
                        res = predict_and_diagnose(row.to_dict())
                        anomali_params = ", ".join(a['Parameter'] for a in res['anomali']) if res['anomali'] else "-"
                        results.append({
                            'Prediksi':           res['prediksi'],
                            'Prob. Layak':        f"{res['prob_layak']:.4f}",
                            'Prob. Tidak Layak':  f"{res['prob_tidak']:.4f}",
                            'Parameter Anomali':  anomali_params,
                        })

                    df_result = pd.concat([df_input[FEATURES].reset_index(drop=True),
                                           pd.DataFrame(results)], axis=1)

                # ── Summary metrics ──
                n_layak  = (df_input.assign(pred=[r['Prediksi'] for r in results])['pred'] == 'LAYAK').sum()
                n_tidak  = len(results) - n_layak

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Sampel", len(results))
                c2.metric("✅ Layak", n_layak, f"{n_layak/len(results):.1%}")
                c3.metric("⛔ Tidak Layak", n_tidak, f"{n_tidak/len(results):.1%}")

                st.markdown('<div class="section-label" style="margin-top:1rem;">Hasil Prediksi</div>', unsafe_allow_html=True)
                st.dataframe(df_result, use_container_width=True, hide_index=True)

                # Download hasil
                csv_result = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇ Download Hasil (CSV)",
                    data=csv_result,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
