"""
AppleOrange Dx — Klasifikasi Apple vs Orange
Deployment model: CNN Custom
Nada Thahira Sosa — 2601
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AppleOrange Dx — Klasifikasi Apple vs Orange",
    page_icon="❖",
    layout="centered",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (128, 128)
MODEL_PATH = "custom_cnn_model.h5"
CLASS_NAMES = {0: "Apple", 1: "Orange"}  # sesuaikan urutan label dengan training
LOW_CONFIDENCE_THRESHOLD = 0.65
MIN_FRUIT_COLOR_RATIO = 0.15

FRUIT_INFO = {
    "Apple": {
        "nama": "Apple (Apel)",
        "warna_khas": "Merah cerah atau hijau, tergantung varietasnya",
        "ciri": "Bentuk bulat dengan sedikit lekukan di bagian atas (tempat tangkai), kulit cenderung mengilap dan halus.",
        "tier": "apple",
    },
    "Orange": {
        "nama": "Orange (Jeruk)",
        "warna_khas": "Oranye pekat merata di seluruh permukaan",
        "ciri": "Bentuk bulat dengan tekstur kulit berpori (dimpled/bertekstur), warna oranye konsisten di semua sisi.",
        "tier": "orange",
    },
}

MODEL_INFO = {
    "accuracy": 0.9313,
    "f1": 0.9308,
    "params": "157,473",
    "epochs": 47,
}

VERSION_LOG = [
    {"versi": "v1.0", "tanggal": "-", "perubahan": "Rilis awal: unggah gambar + prediksi Custom CNN, tampilan dasar Streamlit."},
    {"versi": "v2.0", "tanggal": "-", "perubahan": "UI baru dengan hero header, kartu hasil ramah-pengguna, confidence bar, dan halaman Tentang Model terpisah."},
    {"versi": "v3.0", "tanggal": "-", "perubahan": "Hapus opsi kamera (fokus upload), tambah penjelasan alasan prediksi berbasis analisis warna dominan gambar, dukungan dark mode."},
    {"versi": "v4.0", "tanggal": "-", "perubahan": "Desain ulang total mengikuti pola UI/UX AppleOrange Dx: tema terang/gelap eksplisit, indikator langkah 1-2, sidebar navigasi Diagnosis/Tentang Aplikasi, kartu hasil dan rincian keyakinan per kelas."},
    {"versi": "v5.0", "tanggal": "-", "perubahan": "Hapus seluruh ikon emotikon (diganti simbol formal), ganti label navigasi Diagnosis menjadi Recognition dengan sorotan warna lebih gelap saat aktif, bahasa dibuat lebih formal, tipografi diperbesar dan ditebalkan, serta teks konfirmasi foto disederhanakan."},
    {"versi": "v6.0 (final)", "tanggal": "-", "perubahan": "Selaraskan UI/UX dengan pola TomaLeaf Dx: sidebar tanpa ikon dengan navigasi bertanda '›', kartu konfirmasi foto yang lebih ringkas, dan pengecekan warna dominan gambar (indikasi buah) sebelum gambar dikirim ke model. Perbaiki aturan warna teks pada kondisi hover di sidebar yang sebelumnya membuat teks putih tak terlihat di atas latar terang."},
]

# ----------------------------------------------------------------------------
# TEMA (light & dark didefinisikan eksplisit, tidak bergantung tema bawaan Streamlit)
# ----------------------------------------------------------------------------
THEMES = {
    "light": {
        "bg": "#FBF7F0", "card": "#FFFFFF", "text": "#2A211C", "muted": "#7A6A5E",
        "border": "#EFE2D3", "primary": "#B3451D", "primary_dark": "#6E2A11", "primary_text": "#FBF7F0",
        "input_bg": "#FFFFFF", "track": "#F1E6D8",
        "apple": "#DC2626", "orange": "#EA580C",
    },
    "dark": {
        "bg": "#1C1613", "card": "#28201B", "text": "#F3E9DF", "muted": "#C4B2A2",
        "border": "#40332A", "primary": "#E08A4E", "primary_dark": "#B3672F", "primary_text": "#1C1613",
        "input_bg": "#231C18", "track": "#3A2E26",
        "apple": "#F87171", "orange": "#FB923C",
    },
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "page" not in st.session_state:
    st.session_state.page = "diagnosis"

t = THEMES["dark"] if st.session_state.dark_mode else THEMES["light"]

# ----------------------------------------------------------------------------
# STYLING — semua warna eksplisit, tidak mewarisi warna default Streamlit
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, div {{ font-family: 'Work Sans', sans-serif; }}
    h1, h2, h3, .hero-title {{ font-family: 'Fraunces', serif; }}

    .stApp {{ background: {t['bg']} !important; }}
    .block-container {{ padding-top: 2rem; }}

    /* Paksa semua teks umum ikut warna tema kita */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] {{
        color: {t['text']} !important;
    }}

    .hero {{ text-align: center; padding: 0.6rem 1rem 0.4rem 1rem; }}
    .hero-title {{ font-size: 2.1rem; font-weight: 800; margin: 0.3rem 0 0.1rem 0; color: {t['primary']} !important; }}
    .hero-tagline {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase; color: {t['muted']} !important; margin-bottom: 0.5rem; }}
    .hero-sub {{ font-size: 0.98rem; font-weight: 500; color: {t['muted']} !important; max-width: 480px; margin: 0 auto; line-height: 1.55; }}

    .steps {{ display: flex; justify-content: center; gap: 0.5rem; margin: 1.2rem 0 1.4rem 0; flex-wrap: wrap; }}
    .step {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: {t['muted']} !important;
             padding: 0.4rem 0.9rem; border-radius: 20px; background: {t['track']}; }}
    .step.active {{ background: {t['primary']}; color: {t['primary_text']} !important; font-weight: 600; }}
    .step-num {{ width: 20px; height: 20px; border-radius: 50%; background: rgba(127,127,127,0.25);
                 display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; }}

    .card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 14px;
             padding: 1.3rem 1.5rem; margin-bottom: 1rem; color: {t['text']} !important; font-size: 1rem; }}
    .card b {{ color: {t['text']} !important; font-weight: 800; font-size: 1.05rem; }}
    .card p, .card li {{ color: {t['text']} !important; font-weight: 500; }}
    .tips-list {{ font-size: 0.95rem; margin: 0.4rem 0 0 0; padding-left: 1.1rem; }}
    .tips-list li {{ margin-bottom: 0.3rem; }}

    .result-card {{ border-radius: 14px; padding: 1.5rem 1.6rem; margin: 0.4rem 0 1rem 0; text-align: center; }}
    .result-label {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.5px; opacity: 0.95; text-transform: uppercase; }}
    .result-name {{ font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 800; margin: 0.3rem 0; }}
    .result-conf {{ font-size: 1.05rem; font-weight: 600; opacity: 0.97; }}

    .barrow {{ display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }}
    .barrow-label {{ width: 100px; font-size: 0.8rem; color: {t['text']} !important; flex-shrink: 0; }}
    .barrow-track {{ flex: 1; background: {t['track']}; border-radius: 6px; height: 9px; overflow: hidden; }}
    .barrow-fill {{ height: 100%; border-radius: 6px; }}
    .barrow-pct {{ width: 44px; text-align: right; font-size: 0.78rem; color: {t['text']} !important; }}

    .version-row {{ border-left: 3px solid {t['primary']}; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 1rem; }}
    .version-tag {{ display: inline-block; background: {t['primary']}; color: {t['primary_text']} !important;
                     font-size: 0.76rem; padding: 0.12rem 0.6rem; border-radius: 20px; margin-right: 0.5rem; }}
    .version-date {{ color: {t['muted']} !important; font-size: 0.8rem; }}

    /* Komponen native Streamlit: uploader, tombol, expander */
    [data-testid="stFileUploaderDropzone"] {{
        background: {t['input_bg']} !important; border: 2px dashed {t['border']} !important; border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {t['text']} !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] span {{ font-weight: 700 !important; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: {t['primary']} !important; color: {t['primary_text']} !important; border: none !important;
        font-weight: 700 !important; border-radius: 8px !important;
    }}
    .stButton button, .stDownloadButton button {{
        background: {t['primary']} !important; color: {t['primary_text']} !important;
        border: none !important; border-radius: 10px !important;
    }}
    [data-testid="stExpander"] {{ background: {t['card']} !important; border: 1px solid {t['border']} !important; border-radius: 12px !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {t['text']} !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div {{ font-weight: 700 !important; font-size: 1.05rem !important; }}
    [data-testid="stExpander"] p, [data-testid="stExpander"] li, [data-testid="stExpander"] span {{ color: {t['text']} !important; }}

    /* Perkuat kontras teks tombol (beberapa versi Streamlit bungkus label di elemen anak) */
    .stButton button p, .stButton button div, .stButton button span {{ color: {t['primary_text']} !important; font-weight: 600 !important; }}

    .scroll-box {{ max-height: 380px; overflow-y: auto; padding-right: 6px; }}

    .preview-wrap img {{ border-radius: 12px; }}

    /* Sidebar navigasi */
    section[data-testid="stSidebar"] {{ background: {t['card']} !important; border-right: 1px solid {t['border']}; }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div.sidebar-brand {{ color: {t['text']} !important; }}
    .sidebar-brand {{ font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 800; padding: 0.3rem 0 1rem 0; }}
    section[data-testid="stSidebar"] hr {{ border-color: {t['border']} !important; border-top: 1px solid {t['border']} !important; opacity: 1 !important; margin: 1rem 0 !important; }}

    /* Tombol nav default (tidak aktif): transparan, teks ikut warna tema */
    section[data-testid="stSidebar"] .stButton button {{
        background: transparent !important; color: {t['text']} !important; border: none !important;
        text-align: left !important; justify-content: flex-start !important; font-weight: 500 !important;
        padding: 0.5rem 0.7rem !important; border-radius: 8px !important; box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] .stButton button p {{ color: {t['text']} !important; font-weight: 500 !important; text-align: left !important; }}

    /* Hover pada tombol nav tidak aktif: hanya ganti background, teks TETAP warna tema (bukan putih) */
    section[data-testid="stSidebar"] .stButton button:hover {{ background: {t['track']} !important; }}
    section[data-testid="stSidebar"] .stButton button:hover p,
    section[data-testid="stSidebar"] .stButton button:hover div,
    section[data-testid="stSidebar"] .stButton button:hover span {{ color: {t['text']} !important; }}

    /* Tombol nav aktif (primary): background warna utama, teks kontras terhadap warna utama itu */
    section[data-testid="stSidebar"] button[kind="primary"] {{ background: {t['primary_dark']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"] p,
    section[data-testid="stSidebar"] button[kind="primary"] div,
    section[data-testid="stSidebar"] button[kind="primary"] span {{ color: {t['primary_text']} !important; font-weight: 700 !important; }}

    /* Hover pada tombol nav aktif: tetap kontras, tidak ikut jadi putih-di-atas-terang */
    section[data-testid="stSidebar"] button[kind="primary"]:hover {{ background: {t['primary']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"]:hover p,
    section[data-testid="stSidebar"] button[kind="primary"]:hover div,
    section[data-testid="stSidebar"] button[kind="primary"]:hover span {{ color: {t['primary_text']} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict(image: Image.Image):
    model = load_model()
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    prob_orange = float(model.predict(batch, verbose=0)[0][0])
    return {"Apple": 1 - prob_orange, "Orange": prob_orange}


def is_fruit_like(image: Image.Image) -> tuple[bool, float]:
    """Cek cepat berbasis warna: apakah gambar didominasi warna khas apel/jeruk
    (merah, hijau, oranye, kuning) sebelum dikirim ke model. Tidak menyentuh model sama sekali."""
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Merah (apel): R dominan dibanding G dan B
    red_mask = (r > g * 1.15) & (r > b * 1.15)
    # Hijau (apel varietas hijau): G dominan dibanding R dan B
    green_mask = (g > r * 1.05) & (g > b * 1.05)
    # Oranye/kuning (jeruk): R & G tinggi, B relatif rendah
    orange_mask = (r > b * 1.15) & (g > b * 1.0) & (r > 60) & (r < 250)

    fruit_mask = red_mask | green_mask | orange_mask
    ratio = float(np.mean(fruit_mask))
    return ratio >= MIN_FRUIT_COLOR_RATIO, ratio


def analyze_color_profile(image: Image.Image) -> dict:
    """Analisis warna dominan gambar sebagai indikator visual pendukung prediksi."""
    small = image.convert("RGB").resize((64, 64))
    hsv = np.array(small.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    mask = (sat > 0.25) & (val > 0.25)
    if mask.sum() < 20:
        mask = np.ones_like(sat, dtype=bool)

    mean_hue = float(hue[mask].mean())

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

    return {"hue": mean_hue, "label": label, "desc": desc}


def build_reasoning_text(label: str, confidence: float, color_info: dict) -> str:
    matches_apple = color_info["label"] in ("merah", "hijau")
    matches_orange = color_info["label"] in ("oranye", "kuning-oranye")

    if label == "Apple" and matches_apple:
        support = "Ini konsisten dengan prediksi model — apel memang biasanya memiliki warna " + color_info["desc"] + "."
    elif label == "Orange" and matches_orange:
        support = "Ini konsisten dengan prediksi model — jeruk memang biasanya memiliki warna " + color_info["desc"] + "."
    else:
        support = (
            "Warna saja tidak sepenuhnya menjelaskan prediksi ini — model kemungkinan juga "
            "mempertimbangkan tekstur, bentuk, dan pola permukaan yang tidak terlihat lewat "
            "analisis warna sederhana."
        )

    return (
        f"Warna dominan pada gambar cenderung **{color_info['label']}** ({color_info['desc']}). "
        f"Model memprediksi **{label}** dengan keyakinan **{confidence:.0%}**. {support}"
    )


def render_steps(active: int):
    labels = ["Upload & Konfirmasi", "Lihat Hasil"]
    html = '<div class="steps">'
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == active else ""
        html += f'<div class="step {cls}"><span class="step-num">{i}</span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HALAMAN: DIAGNOSIS (KLASIFIKASI)
# ----------------------------------------------------------------------------
def render_diagnosis():
    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size:1.9rem; font-weight:800; color:{t['primary']};">❖</div>
            <div class="hero-title">AppleOrange Dx</div>
            <div class="hero-tagline">Klasifikasi Apple vs Orange</div>
            <div class="hero-sub">Unggah foto buah Anda untuk mengetahui apakah itu apel atau jeruk,
            lengkap dengan alasan di balik prediksinya.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.markdown(
            f"""
            <div class="card">
            File model <code>{MODEL_PATH}</code> tidak ditemukan. Pastikan file .h5 hasil
            training berada di root repo, sejajar dengan app.py.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if "stage" not in st.session_state:
        st.session_state.stage = 1
    if "probs" not in st.session_state:
        st.session_state.probs = None
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None

    render_steps(st.session_state.stage)

    if st.session_state.stage == 1:
        uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded is None:
            st.markdown(
                """
                <div class="card">
                <b>Panduan Penggunaan</b>
                <ul class="tips-list">
                    <li>Klik kotak di atas, atau tarik dan lepas foto buah.</li>
                    <li>Gunakan foto <b>close-up satu buah</b> dengan pencahayaan cukup dan latar polos.</li>
                    <li>Konfirmasi foto terlebih dahulu, kemudian sistem akan mengklasifikasikan dan menjelaskan alasannya.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded)

            col_l, col_mid, col_r = st.columns([1, 2, 1])
            with col_mid:
                st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption="Preview foto")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="card" style="text-align:center;">
                <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">Konfirmasi Foto</div>
                <div>Periksa foto sebelum memulai klasifikasi.<br>
                Pastikan buah terlihat jelas dan fokus. Jika ingin mengganti foto, klik tombol × di bagian atas.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Mulai Klasifikasi →", type="primary", use_container_width=True):
                fruit_like, ratio = is_fruit_like(image)
                if not fruit_like:
                    st.error(
                        f"Gambar tidak terdeteksi sebagai buah apel/jeruk (kecocokan warna khas hanya {ratio*100:.0f}%). "
                        "Coba unggah foto close-up satu buah dengan pencahayaan yang cukup."
                    )
                else:
                    with st.spinner("Sedang menganalisis buah..."):
                        probs = predict(image)
                        color_info = analyze_color_profile(image)
                    st.session_state.image_bytes = uploaded.getvalue()
                    st.session_state.probs = probs
                    st.session_state.color_info = color_info
                    st.session_state.stage = 2
                    st.rerun()

    elif st.session_state.stage == 2:
        probs = st.session_state.probs
        top_label = max(probs, key=probs.get)
        confidence = probs[top_label]
        info = FRUIT_INFO[top_label]
        color = t[info["tier"]]

        st.markdown(
            f"""
            <div class="result-card" style="background:{color}; color:#FFFFFF;">
                <div class="result-label">Hasil Klasifikasi</div>
                <div class="result-name">{info['nama']}</div>
                <div class="result-conf">Tingkat keyakinan model: {confidence*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.markdown(
                """
                <div class="card">
                Keyakinan model cukup rendah untuk gambar ini. Coba gunakan foto dengan
                pencahayaan lebih jelas dan latar belakang polos untuk hasil yang lebih akurat.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="card">
                <p style="margin:0 0 0.4rem 0;"><b>Warna khas:</b> {info['warna_khas']}</p>
                <p style="margin:0;"><b>Ciri bentuk & tekstur:</b> {info['ciri']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Kenapa model bilang begitu?", expanded=True):
            reasoning = build_reasoning_text(top_label, confidence, st.session_state.color_info)
            st.write(reasoning)
            st.caption(
                "Catatan: ini adalah indikator visual pendukung berbasis warna dominan gambar, "
                "bukan pembongkaran langsung isi \"otak\" model. Model CNN sebenarnya belajar "
                "dari kombinasi warna, tekstur, dan bentuk sekaligus."
            )

        with st.expander("Lihat rincian keyakinan untuk semua kelas"):
            order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            bars = []
            for cls_name, pct_val in order:
                pct = pct_val * 100
                bars.append(
                    f'<div class="barrow">'
                    f'<div class="barrow-label">{FRUIT_INFO[cls_name]["nama"].split(" ")[0]}</div>'
                    f'<div class="barrow-track"><div class="barrow-fill" style="width:{pct:.1f}%; background:{t[FRUIT_INFO[cls_name]["tier"]]};"></div></div>'
                    f'<div class="barrow-pct">{pct:.1f}%</div>'
                    f'</div>'
                )
            bars_html = '<div class="scroll-box">' + "".join(bars) + "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

        if st.button("↻ Unggah Foto Lain", use_container_width=True):
            st.session_state.stage = 1
            st.session_state.image_bytes = None
            st.session_state.probs = None
            st.rerun()

    st.caption("AppleOrange Dx · Model: CNN Custom · Nada Thahira Sosa — 2601")


# ----------------------------------------------------------------------------
# HALAMAN: TENTANG APLIKASI
# ----------------------------------------------------------------------------
def render_about():
    st.markdown('<div class="hero-title" style="text-align:left; font-size:1.6rem; margin-bottom:1rem;">Tentang Aplikasi</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <b>Tentang AppleOrange Dx</b>
            <p style="margin:0.5rem 0 0.4rem 0;">AppleOrange Dx merupakan aplikasi untuk membantu
            mengidentifikasi apakah sebuah foto buah adalah apel atau jeruk.</p>
            <p style="margin:0;">Aplikasi ini memberikan alasan prediksi berbasis analisis warna
            dominan gambar, agar hasil klasifikasi lebih mudah dipahami.</p>
        </div>

        <div class="card">
            <b>Model yang Digunakan</b>
            <p style="margin:0.5rem 0 0.4rem 0; font-weight:600;">Custom CNN (dibangun dari nol)</p>
            <p style="margin:0;">Model dilatih untuk membedakan Apple dan Orange dari citra 128×128
            piksel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
    m2.metric("F1 Score", f"{MODEL_INFO['f1']:.4f}")
    m3.metric("Total Params", MODEL_INFO["params"])
    m4.metric("Epochs", MODEL_INFO["epochs"])

    with st.expander("Riwayat Versi", expanded=False):
        rows = []
        for v in VERSION_LOG:
            rows.append(
                f'<div class="version-row">'
                f'<span class="version-tag">{v["versi"]}</span>'
                f'<span class="version-date">{v["tanggal"]}</span>'
                f'<p style="margin:0.3rem 0 0 0;">{v["perubahan"]}</p>'
                f'</div>'
            )
        version_html = '<div class="scroll-box">' + "".join(rows) + "</div>"
        st.markdown(version_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <b>Informasi Aplikasi</b>
            <p style="margin:0.6rem 0 0 0;">AppleOrange Dx<br>Model: CNN Custom<br>Nada Thahira Sosa — 2601</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR — NAVIGASI UTAMA
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">AppleOrange Dx</div>', unsafe_allow_html=True)

    if st.button("› Recognition", key="nav_diagnosis", use_container_width=True,
                 type="primary" if st.session_state.page == "diagnosis" else "secondary"):
        st.session_state.page = "diagnosis"
        st.rerun()

    if st.button("› Tentang Aplikasi", key="nav_about", use_container_width=True,
                 type="primary" if st.session_state.page == "about" else "secondary"):
        st.session_state.page = "about"
        st.rerun()

    st.markdown("---")
    theme_label = "Mode: Gelap" if st.session_state.dark_mode else "Mode: Terang"
    if st.button(f"◐ {theme_label}", key="theme_toggle_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
if st.session_state.page == "diagnosis":
    render_diagnosis()
else:
    render_about()
