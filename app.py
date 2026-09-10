"""
Apple vs Orange Classifier — Streamlit Deployment App
Author  : Nada Thahira Sosa (2601)
Project : LAS26 Case 1 — Week 2/3/4 Big Data / Machine Learning
Models  : Custom CNN (from scratch) & MobileNetV2 (transfer learning)

Run locally:
    streamlit run app.py

Required files in the same folder:
    - custom_cnn_model.h5
    - pretrained_mobilenetv2_model.h5
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple vs Orange Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 128
MODEL_FILES = {
    "Custom CNN": "custom_cnn_model.h5",
    "MobileNetV2 (Transfer Learning)": "pretrained_mobilenetv2_model.h5",
}
MODEL_METRICS = {
    "Custom CNN": {
        "accuracy": 0.9313,
        "f1": 0.9308,
        "params": "157,473",
        "trainable": "156,769",
        "epochs": 47,
        "color": "#EF4444",
        "note": "Dibangun dari nol (VGG-style 3x3 conv blocks). Ringan, self-contained, "
                "tidak butuh koneksi internet untuk bobot pretrained.",
    },
    "MobileNetV2 (Transfer Learning)": {
        "accuracy": 0.9563,
        "f1": 0.9554,
        "params": "2,422,593",
        "trainable": "164,353",
        "epochs": 94,
        "color": "#F97316",
        "note": "Backbone ImageNet di-freeze, hanya classifier head yang dilatih. "
                "Akurasi tertinggi & konvergen lebih cepat ke performa tinggi.",
    },
}
CLASS_NAMES = ["Apple", "Orange"]
CLASS_EMOJI = {"Apple": "🍎", "Orange": "🍊"}
CLASS_COLOR = {"Apple": "#EF4444", "Orange": "#F97316"}

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .hero {
        background: linear-gradient(120deg, #EF4444 0%, #F97316 55%, #FACC15 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero p  { margin: 0.4rem 0 0 0; font-size: 1rem; opacity: 0.95; }

    .card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin-right: 0.4rem;
    }

    .result-badge {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }

    .confidence-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: #F1F5F9;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] {
        background: #0F172A;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label { 
        color: #E2E8F0 !important; 
    }

    /* Style file uploader agar bersih, jelas, dan kontras rapi */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #F97316 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #475569 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #475569 !important;
    }
    /* Chip file yang sudah diunggah */
    [data-testid="stFileUploaderFileData"] {
        background-color: #F1F5F9 !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 0.4rem 0.6rem !important;
    }
    [data-testid="stFileUploaderFileData"] * {
        color: #1E293B !important;
    }

    /* Outer feature card styling */
    .feature-box {
        background: white;
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }
    .feature-box-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .subcard {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        border: 1px solid #E2E8F0;
        height: 100%;
    }
    .subcard-tag {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #C2410C;
        margin-bottom: 0.25rem;
    }
    .subcard-subtitle {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .subcard-desc {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.45;
        margin: 0;
    }
    .image-preview-container {
        display: flex;
        justify-content: center;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# MODEL LOADING (cached — optimisation: avoids re-loading weights on every rerun)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not Path(path).exists():
        return None
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data(show_spinner=False)
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize to 128x128 and scale to [0,1], matching the notebook's preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict(model, batch: np.ndarray):
    prob_orange = float(model.predict(batch, verbose=0)[0][0])
    label = "Orange" if prob_orange > 0.5 else "Apple"
    confidence = prob_orange if prob_orange > 0.5 else 1 - prob_orange
    return label, confidence, prob_orange


# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍏 Menu")
    mode = st.radio(
        "Mode prediksi",
        ["Bandingkan kedua model", "Custom CNN saja", "MobileNetV2 saja"],
        index=0,
    )
    st.divider()
    threshold = st.slider(
        "Ambang keputusan (decision threshold)",
        min_value=0.30, max_value=0.70, value=0.50, step=0.01,
        help="Probabilitas > threshold → diprediksi sebagai Orange. Geser untuk melihat "
             "efek trade-off precision/recall secara interaktif (fitur eksplorasi tambahan).",
    )
    st.divider()
    st.markdown("### ℹ️ Tentang Proyek")
    st.caption(
        "Klasifikasi citra biner **Apple vs Orange** menggunakan dua pendekatan: "
        "Custom CNN (from scratch) dan MobileNetV2 (transfer learning), "
        "dilatih pada dataset 796 gambar 128×128."
    )
    st.markdown("**Dibuat oleh:** Nada Thahira Sosa · Kode CaAs 2601")

# --------------------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🍎 Apple vs Orange Classifier 🍊</h1>
        <p>Unggah foto apel atau jeruk, lalu bandingkan hasil prediksi Custom CNN vs MobileNetV2 secara langsung.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_compare, tab_about = st.tabs(["🔍 Prediksi", "📊 Perbandingan Model", "📄 Tentang & Versi"])

# --------------------------------------------------------------------------------------
# TAB 1 — PREDICT
# --------------------------------------------------------------------------------------
with tab_predict:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-box-title'>📤 1. Unggah Gambar</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Format didukung: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            # Preview lebih ringkas (max 220px) agar muat dalam 1 layar bersama tombol klasifikasi
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p2:
                st.image(uploaded, caption="Konfirmasi Foto", width=200)
            
            btn_classify = st.button("🚀 Mulai Klasifikasi", type="primary", use_container_width=True)
        else:
            st.info("Unggah gambar apel atau jeruk untuk memulai konfirmasi dan prediksi.")
            btn_classify = False
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-box-title'>🎯 2. Hasil Prediksi</div>", unsafe_allow_html=True)

        if uploaded is not None:
            if "has_classified" not in st.session_state:
                st.session_state.has_classified = False
            if btn_classify:
                st.session_state.has_classified = True

            image_bytes = uploaded.getvalue()
            batch, pil_img = preprocess_image(image_bytes)

            active_models = []
            if mode in ("Bandingkan kedua model", "Custom CNN saja"):
                active_models.append("Custom CNN")
            if mode in ("Bandingkan kedua model", "MobileNetV2 saja"):
                active_models.append("MobileNetV2 (Transfer Learning)")

            predicted_labels = []
            for name in active_models:
                model = load_model(MODEL_FILES[name])
                info = MODEL_METRICS[name]

                st.markdown(f"<div class='card' style='box-shadow:none; border:1px solid #E2E8F0; margin-bottom:0.8rem;'>", unsafe_allow_html=True)
                st.markdown(f"**{name}**")

                if model is None:
                    st.warning(
                        f"File model `{MODEL_FILES[name]}` tidak ditemukan di folder deploy. "
                        "Pastikan file .h5 hasil training berada di direktori yang sama dengan app.py."
                    )
                else:
                    t0 = time.time()
                    prob_orange = float(model.predict(batch, verbose=0)[0][0])
                    latency = (time.time() - t0) * 1000
                    label = "Orange" if prob_orange > threshold else "Apple"
                    confidence = prob_orange if label == "Orange" else 1 - prob_orange
                    color = CLASS_COLOR[label]
                    predicted_labels.append((label, confidence))

                    st.markdown(
                        f"<div class='result-badge' style='color:{color}; font-size:2.1rem;'>"
                        f"{CLASS_EMOJI[label]} {label}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='confidence-track'>"
                        f"<div class='confidence-fill' style='width:{confidence*100:.1f}%; background:{color};'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Keyakinan model: **{confidence*100:.2f}%** · waktu inferensi ≈ {latency:.0f} ms")

                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#64748B; font-size:0.9rem; padding:1.5rem; text-align:center;'>"
                "Belum ada gambar yang diunggah. Silakan unggah foto pada panel di sebelah kiri.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Box Karakteristik Visual & Interpretasi (Judul berada di dalam kotak)
    if uploaded is not None:
        main_label = predicted_labels[0][0] if predicted_labels else "Apple"
        main_conf = predicted_labels[0][1] if predicted_labels else 0.9

        st.markdown(
            """
            <div class="feature-box">
                <div class="feature-box-title">🔍 Karakteristik Visual & Analisis Fitur</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div class="subcard">
                        <div class="subcard-tag">◆ WARNA</div>
                        <div class="subcard-subtitle">Merah • Hijau • Kuning / Oranye</div>
                        <p class="subcard-desc">Distribusi warna pada citra dianalisis oleh filter konvolusi. Variasi rona dapat dipengaruhi oleh kematangan buah dan pencahayaan foto.</p>
                    </div>
                    <div class="subcard">
                        <div class="subcard-tag">■ BENTUK</div>
                        <div class="subcard-subtitle">Bulat • Agak lonjong / Simetris</div>
                        <p class="subcard-desc">Kontur tepi dan proporsi geometri buah memberikan fitur spasial yang membedakan struktur lekukan apel dengan kebulatan jeruk.</p>
                    </div>
                    <div class="subcard" style="grid-column: 1 / -1;">
                        <div class="subcard-tag">▲ TEKSTUR PERMUKAAN</div>
                        <div class="subcard-subtitle">Halus • Mengilap • Sedikit berpori / berlilin</div>
                        <p class="subcard-desc">Kondisi permukaan kulit dan pantulan kilau cahaya membantu feature extractor mengenali pori khas kulit jeruk atau lapisan lilin apel.</p>
                    </div>
                </div>
                <div class="subcard" style="background:#FFFFFF; border-left: 4px solid #F97316;">
                    <div class="subcard-tag" style="color:#0F172A; font-size:0.8rem;">§ INTERPRETASI PREDIKSI</div>
                    <p class="subcard-desc" style="font-size:0.85rem; color:#334155;">
                        Tingkat keyakinan model tergolong tinggi. Fitur visual yang diekstraksi dari bobot konvolusi selaras dengan karakteristik kelas yang diprediksi. Pemeriksaan silang tetap disarankan untuk citra dengan pencahayaan ekstrem atau sudut pandang yang tidak umum.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------------------
# TAB 2 — MODEL COMPARISON (static, from notebook evaluation)
# --------------------------------------------------------------------------------------
with tab_compare:
    st.markdown("#### Performa pada Test Set (160 gambar, 80 Apple + 80 Orange)")
    c1, c2 = st.columns(2, gap="large")

    for col, name in zip([c1, c2], MODEL_METRICS.keys()):
        info = MODEL_METRICS[name]
        with col:
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='metric-pill' style='background:{info['color']}'>{name}</span>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{info['accuracy']*100:.2f}%")
            m2.metric("F1 Score", f"{info['f1']:.4f}")
            m3, m4 = st.columns(2)
            m3.metric("Total Params", info["params"])
            m4.metric("Trainable Params", info["trainable"])
            st.caption(f"Epochs sampai early-stopping: **{info['epochs']}**")
            st.write(info["note"])
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
        <b>Kesimpulan:</b> MobileNetV2 (Transfer Learning) unggul pada kedua metrik utama
        (Accuracy 95,63% vs 93,13%; F1 Score 0,9554 vs 0,9308) dan menjadi <b>model terbaik</b>
        yang direkomendasikan untuk produksi, meskipun Custom CNN tetap kompetitif dengan
        ukuran model ~15× lebih kecil — cocok jika prioritasnya adalah efisiensi deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# TAB 3 — ABOUT & VERSIONING
# --------------------------------------------------------------------------------------
with tab_about:
    st.markdown("#### Tentang Aplikasi")
    st.write(
        "Aplikasi ini men-deploy dua model klasifikasi citra biner (Apple vs Orange) yang "
        "dikembangkan pada Week 2–4: **Custom CNN** (dibangun dari nol) dan **MobileNetV2** "
        "(transfer learning dari ImageNet). Pengguna dapat mengunggah gambar, memilih model "
        "yang ingin digunakan, dan membandingkan hasil prediksi keduanya secara langsung."
    )

    st.markdown("#### 🧾 Riwayat Versi (Versioning)")
    st.table(
        {
            "Versi": ["v1.0", "v2.0", "v2.1 (Final)"],
            "Tanggal": ["-", "-", "-"],
            "Perubahan": [
                "Rilis awal: unggah gambar + prediksi satu model (MobileNetV2), tampilan dasar Streamlit.",
                "Menambahkan mode perbandingan dua model sekaligus, tab Perbandingan Model dengan metrik "
                "kuantitatif, dan pewarnaan hasil per kelas.",
                "Menambahkan slider decision threshold interaktif, caching model (optimisasi kecepatan), "
                "indikator waktu inferensi, dan penyempurnaan UI (hero header, kartu, progress bar keyakinan).",
            ],
            "Screenshot": ["_(lampirkan tangkapan layar di sini)_"] * 3,
        }
    )
    st.caption(
        "Catatan: isi kolom Tanggal dan Screenshot sesuai riwayat deployment Anda yang sebenarnya "
        "(mis. tangkapan layar Streamlit Cloud pada tiap versi)."
    )

    st.markdown("#### 📦 Struktur File Deployment")
    st.code(
        "project/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "├── custom_cnn_model.h5\n"
        "└── pretrained_mobilenetv2_model.h5",
        language="text",
    )

st.markdown(
    "<div class='footer-note'>Dibangun dengan Streamlit · TensorFlow/Keras · "
    "Apple vs Orange Binary Classification Project</div>",
    unsafe_allow_html=True,
)
