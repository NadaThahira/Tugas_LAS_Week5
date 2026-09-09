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
        "text": "#1F2937",
    },
    "Orange": {
        "emoji": "🍊",
        "headline": "Ini terlihat seperti Jeruk",
        "color": "#EA580C",
        "bg": "#FFF7ED",
        "text": "#1F2937",
    },
}

st.set_page_config(page_title="Klasifikasi Apple vs Orange", page_icon="🍎🍊", layout="centered")

# --------------------------------------------------------------------------------------
# STYLE — pakai CSS variable bawaan Streamlit supaya otomatis ikut light/dark mode
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .hero { text-align: center; padding: 0.6rem 0 1.4rem 0; }
    .hero .icon { font-size: 2.6rem; }
    .hero h1 {
        margin: 0.3rem 0 0.2rem 0;
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--text-color);
    }
    .hero p  { margin: 0; color: var(--text-color); opacity: 0.65; font-size: 0.98rem; }

    .result-card {
        text-align: center;
        border-radius: 22px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1rem;
    }
    .result-emoji { font-size: 3rem; margin-bottom: 0.2rem; }
    .result-headline { font-size: 1.4rem; font-weight: 800; margin: 0.2rem 0 0.3rem 0; }
    .result-sub { font-size: 1rem; opacity: 0.85; margin-bottom: 0.9rem; }

    .confidence-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: rgba(120, 120, 120, 0.25);
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
        color: var(--text-color);
        opacity: 0.45;
        font-size: 0.8rem;
        margin-top: 1.6rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(120, 120, 120, 0.25);
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


def _find_last_conv_layer(model):
    """Cari layer Conv2D terakhir secara otomatis, tidak bergantung pada nama layer."""
    last_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_name = layer.name
    return last_name


def analyze_color_profile(image: Image.Image) -> dict:
    """Analisis warna dominan gambar sebagai indikator visual pendukung prediksi."""
    small = image.convert("RGB").resize((64, 64))
    hsv = np.array(small.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)  # PIL HSV hue 0-255 -> derajat 0-360
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    # Hanya piksel yang cukup jenuh & terang dilibatkan (buang bayangan/latar netral)
    mask = (sat > 0.25) & (val > 0.25)
    if mask.sum() < 20:
        mask = np.ones_like(sat, dtype=bool)

    mean_hue = float(hue[mask].mean())
    mean_sat = float(sat[mask].mean())

    if mean_hue < 15 or mean_hue >= 345:
        label, desc = "merah", "merah cerah khas apel"
    elif mean_hue < 45:
        label, desc = "oranye", "oranye pekat khas jeruk"
    elif mean_hue < 70:
        label, desc = "kuning-oranye", "kuning kecokelatan"
    elif mean_hue < 160:
        label, desc = "hijau", "hijau, seperti apel varietas hijau"
    else:
        label, desc = "campuran", "campuran warna yang kurang khas"

    return {"hue": mean_hue, "saturation": mean_sat, "label": label, "desc": desc}


def build_reasoning_text(label: str, confidence: float, color_info: dict) -> str:
    """Susun alasan berbasis kata-kata dari kombinasi prediksi model + analisis warna."""
    matches_apple = color_info["label"] in ("merah", "hijau")
    matches_orange = color_info["label"] in ("oranye", "kuning-oranye")

    if label == "Apple" and matches_apple:
        support = (
            f"Ini konsisten dengan prediksi model — apel memang biasanya memiliki warna "
            f"{color_info['desc']}."
        )
    elif label == "Orange" and matches_orange:
        support = (
            f"Ini konsisten dengan prediksi model — jeruk memang biasanya memiliki warna "
            f"{color_info['desc']}."
        )
    else:
        support = (
            "Warna saja tidak sepenuhnya menjelaskan prediksi ini — model kemungkinan juga "
            "mempertimbangkan tekstur, bentuk, dan pola permukaan yang tidak terlihat lewat "
            "analisis warna sederhana."
        )

    return (
        f"Warna dominan pada gambar cenderung **{color_info['label']}** "
        f"({color_info['desc']}). Model memprediksi **{label}** dengan keyakinan "
        f"**{confidence:.0%}**. {support}"
    )


# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None  # gambar yang baru dipilih, belum dikonfirmasi
if "image" not in st.session_state:
    st.session_state.image = None  # gambar yang sudah dikonfirmasi user
if "result" not in st.session_state:
    st.session_state.result = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5


def reset():
    st.session_state.pending_image = None
    st.session_state.image = None
    st.session_state.result = None


def confirm_image():
    st.session_state.image = st.session_state.pending_image
    st.session_state.pending_image = None


def cancel_pending():
    st.session_state.pending_image = None


# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍎🍊 Menu")
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

    with st.container(border=True):
        st.markdown("**Arsitektur:** Custom CNN (dibangun dari nol)")
        st.write(
            "Model dilatih untuk membedakan **Apple** dan **Orange** dari citra 128×128 piksel."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
        m2.metric("F1 Score", f"{MODEL_INFO['f1']:.4f}")
        m3.metric("Total Params", MODEL_INFO["params"])
        m4.metric("Epochs", MODEL_INFO["epochs"])

    with st.expander("⚙️ Pengaturan lanjutan (opsional)"):
        st.session_state.threshold = st.slider(
            "Ambang keputusan (decision threshold)",
            min_value=0.30, max_value=0.70, value=st.session_state.threshold, step=0.01,
            help="Probabilitas ≥ threshold → diprediksi Orange. Nilai default 0.5 sudah "
                 "optimal untuk sebagian besar kasus.",
        )

    with st.container(border=True):
        st.markdown("**🧾 Riwayat Versi**")
        st.table(
            {
                "Versi": ["v1.0", "v2.0", "v2.1", "v2.2 (Final)"],
                "Tanggal": ["-", "-", "-", "-"],
                "Perubahan": [
                    "Rilis awal: unggah gambar + prediksi Custom CNN, tampilan dasar.",
                    "UI baru dengan hero header, kartu hasil ramah-pengguna (bahasa natural "
                    "bukan sekadar label), confidence bar, opsi ambil foto via kamera, dan "
                    "halaman Tentang Model terpisah untuk detail teknis.",
                    "Menghapus opsi ambil foto via kamera (fokus upload saja), menambahkan "
                    "penjelasan visual Grad-CAM yang menyoroti area gambar paling berpengaruh "
                    "terhadap keputusan model beserta deskripsi lokasinya.",
                    "Perbaikan bug penjelasan yang gagal, ganti kartu manual jadi container "
                    "bawaan Streamlit (mendukung dark mode), tambah langkah konfirmasi foto "
                    "sebelum diproses, logo ganda Apple+Orange, dan alasan prediksi dalam "
                    "bahasa natural berbasis analisis warna dominan.",
                ],
                "Screenshot": ["_(lampirkan di sini)_"] * 4,
            }
        )

# ========================================================================================
# PAGE: KLASIFIKASI
# ========================================================================================
else:
    st.markdown(
        """
        <div class="hero">
            <div class="icon">🍎🍊</div>
            <h1>Klasifikasi Apple vs Orange</h1>
            <p>Unggah foto buahmu untuk mengetahui apakah itu apel atau jeruk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        with st.container(border=True):
            st.warning(
                f"File model `{MODEL_PATH}` tidak ditemukan. Pastikan file .h5 berada di root "
                "repo, sejajar dengan app.py."
            )

    # Layar 1: pilih & konfirmasi gambar
    elif st.session_state.image is None:
        with st.container(border=True):
            st.markdown("**📁 Upload foto buah**")
            uploaded = st.file_uploader(
                "Pilih gambar buah", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )
            if uploaded is not None:
                st.session_state.pending_image = Image.open(uploaded)

            if st.session_state.pending_image is not None:
                st.image(
                    st.session_state.pending_image,
                    caption="Preview — pastikan foto ini sudah benar",
                    use_container_width=True,
                )
                col1, col2 = st.columns(2)
                col1.button(
                    "✅ Gunakan foto ini", type="primary",
                    use_container_width=True, on_click=confirm_image,
                )
                col2.button(
                    "🔄 Batalkan", use_container_width=True, on_click=cancel_pending,
                )
                st.caption("Belum yakin? Pilih file lain di atas untuk mengganti foto.")

    # Layar 2 & 3: klasifikasi, hasil
    else:
        with st.container(border=True):
            st.image(st.session_state.image, caption="Foto buah", use_container_width=True)

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
                <div class="result-card" style="background:{copy['bg']}; color:{copy['text']};">
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

            # ---- Alasan dalam kata-kata (analisis warna) ----
            with st.expander("🔎 Kenapa model bilang begitu?", expanded=True):
                color_info = analyze_color_profile(st.session_state.image)
                st.write(build_reasoning_text(label, confidence, color_info))
                st.caption(
                    "Catatan: ini adalah indikator visual pendukung berbasis warna dominan "
                    "gambar, bukan pembongkaran langsung isi \"otak\" model. Model CNN "
                    "sebenarnya belajar dari kombinasi warna, tekstur, dan bentuk sekaligus, "
                    "sehingga penjelasan lengkapnya lebih kompleks daripada warna saja."
                )

            st.button("↻ Uji gambar lain", on_click=reset, use_container_width=True)

    st.markdown(
        "<div class='footer-note'>Dibangun dengan Streamlit · TensorFlow/Keras</div>",
        unsafe_allow_html=True,
    )
