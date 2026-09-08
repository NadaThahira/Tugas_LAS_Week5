"""
Klasifikasi Apple vs Orange — Streamlit App
Author : Nada Thahira Sosa (2601)
Model  : Custom CNN (from scratch)

Run:
    streamlit run app.py

Required file in the same folder:
    - custom_cnn_model.h5
"""

import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
MODEL_PATH = "custom_cnn_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = {0: "Apple", 1: "Orange"}  # sesuaikan urutan label dengan training
LOW_CONFIDENCE_THRESHOLD = 0.65

MODEL_INFO = {
    "accuracy": 0.9313,
    "f1": 0.9308,
    "params": "157,473",
    "trainable": "156,769",
    "epochs": 47,
}

RESULT_COPY = {
    "Apple": {
        "emoji": "🍎",
        "headline": "Ini terlihat seperti Apel",
        "color": "#DC2626",
        "bg": "#FEF2F2",
    },
    "Orange": {
        "emoji": "🍊",
        "headline": "Ini terlihat seperti Jeruk",
        "color": "#EA580C",
        "bg": "#FFF7ED",
    },
}

st.set_page_config(page_title="Klasifikasi Apple vs Orange", page_icon="🍎", layout="centered")

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .hero { text-align: center; padding: 0.6rem 0 1.4rem 0; }
    .hero .icon { font-size: 2.6rem; }
    .hero h1 { margin: 0.3rem 0 0.2rem 0; font-size: 1.7rem; font-weight: 800; color: #0F172A; }
    .hero p  { margin: 0; color: #64748B; font-size: 0.98rem; }

    .card {
        background: white;
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .result-card {
        text-align: center;
        border-radius: 22px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1rem;
    }
    .result-emoji { font-size: 3rem; margin-bottom: 0.2rem; }
    .result-headline { font-size: 1.4rem; font-weight: 800; margin: 0.2rem 0 0.3rem 0; }
    .result-sub { font-size: 1rem; color: #475569; margin-bottom: 0.9rem; }

    .confidence-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: #E2E8F0;
        overflow: hidden;
    }
    .confidence-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }

    div.stButton > button {
        border-radius: 14px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 1.6rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, threshold: float):
    x = preprocess(image)
    t0 = time.time()
    prob = float(model.predict(x, verbose=0)[0][0])
    latency = (time.time() - t0) * 1000
    label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
    confidence = prob if prob >= threshold else 1 - prob
    return label, confidence, latency


# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None
if "result" not in st.session_state:
    st.session_state.result = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5


def reset():
    st.session_state.image = None
    st.session_state.result = None


# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍏 Menu")
    page = st.radio("Halaman", ["Klasifikasi", "Tentang Model"], label_visibility="collapsed")

model = load_model()

# ========================================================================================
# PAGE: TENTANG MODEL
# ========================================================================================
if page == "Tentang Model":
    st.markdown(
        """
        <div class="hero">
            <div class="icon">📄</div>
            <h1>Tentang Model</h1>
            <p>Detail teknis di balik klasifikasi apple vs orange ini.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Arsitektur:** Custom CNN (dibangun dari nol)")
    st.write(
        "Model dilatih untuk membedakan **Apple** dan **Orange** dari citra 128×128 piksel."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
    m2.metric("F1 Score", f"{MODEL_INFO['f1']:.4f}")
    m3.metric("Total Params", MODEL_INFO["params"])
    m4.metric("Epochs", MODEL_INFO["epochs"])
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Pengaturan lanjutan (opsional)"):
        st.session_state.threshold = st.slider(
            "Ambang keputusan (decision threshold)",
            min_value=0.30, max_value=0.70, value=st.session_state.threshold, step=0.01,
            help="Probabilitas ≥ threshold → diprediksi Orange. Nilai default 0.5 sudah "
                 "optimal untuk sebagian besar kasus.",
        )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**🧾 Riwayat Versi**")
    st.table(
        {
            "Versi": ["v1.0", "v2.0 (Final)"],
            "Tanggal": ["-", "-"],
            "Perubahan": [
                "Rilis awal: unggah/foto gambar + prediksi Custom CNN, tampilan dasar.",
                "UI baru dengan hero header, kartu hasil ramah-pengguna (bahasa natural "
                "bukan sekadar label), confidence bar, opsi ambil foto via kamera, dan "
                "halaman Tentang Model terpisah untuk detail teknis.",
            ],
            "Screenshot": ["_(lampirkan di sini)_"] * 2,
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================================================
# PAGE: KLASIFIKASI
# ========================================================================================
else:
    st.markdown(
        """
        <div class="hero">
            <div class="icon">🍎</div>
            <h1>Klasifikasi Apple vs Orange</h1>
            <p>Unggah atau foto buahmu untuk mengetahui apakah itu apel atau jeruk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.warning(
            f"File model `{MODEL_PATH}` tidak ditemukan. Pastikan file .h5 berada di root "
            "repo, sejajar dengan app.py."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Layar 1: input gambar
    elif st.session_state.image is None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab_camera, tab_upload = st.tabs(["📷 Ambil Foto", "📁 Upload Foto"])

        with tab_camera:
            captured = st.camera_input("Ambil foto buah", label_visibility="collapsed")
            if captured is not None:
                st.session_state.image = Image.open(captured)
                st.rerun()

        with tab_upload:
            uploaded = st.file_uploader(
                "Pilih gambar buah", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )
            if uploaded is not None:
                st.session_state.image = Image.open(uploaded)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Layar 2 & 3: preview, klasifikasi, hasil
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(st.session_state.image, caption="Foto buah", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.result is None:
            if st.button("🔍 Klasifikasikan", type="primary", use_container_width=True):
                with st.spinner("Menganalisis gambar..."):
                    st.session_state.result = predict(
                        model, st.session_state.image, st.session_state.threshold
                    )
                st.rerun()
            st.button("↻ Uji gambar lain", on_click=reset, use_container_width=True)

        else:
            label, confidence, _latency = st.session_state.result
            copy = RESULT_COPY[label]

            st.markdown(
                f"""
                <div class="result-card" style="background:{copy['bg']};">
                    <div class="result-emoji">{copy['emoji']}</div>
                    <div class="result-headline" style="color:{copy['color']};">{copy['headline']}</div>
                    <div class="result-sub">Model yakin {confidence:.0%}</div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width:{confidence*100:.1f}%; background:{copy['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    "Keyakinan model rendah. Coba gunakan foto dengan pencahayaan lebih "
                    "jelas dan latar belakang polos."
                )

            st.button("↻ Uji gambar lain", on_click=reset, use_container_width=True)

    st.markdown(
        "<div class='footer-note'>Dibangun dengan Streamlit · TensorFlow/Keras</div>",
        unsafe_allow_html=True,
    )
