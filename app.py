"""
FruitID — Apple & Orange Recognition
Deployment model: CNN Custom
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
    page_title="FruitID — Apple & Orange Recognition",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (128, 128)
MODEL_PATH = "custom_cnn_model.h5"
CLASS_NAMES = {0: "Apple", 1: "Orange"}  # sesuaikan urutan label dengan training
LOW_CONFIDENCE_THRESHOLD = 0.65

FRUIT_INFO = {
    "Apple": {
        "nama": "Apple (Apel)",
        "warna_khas": (
            "Merah cerah, merah kehijauan, hijau, atau kuning, tergantung varietasnya "
            "(misalnya Fuji dan Red Delicious cenderung merah pekat, sementara Granny Smith "
            "hijau terang dan Golden Delicious kuning keemasan). Warna biasanya tidak rata "
            "sempurna — sering ada semburat/gradasi warna serta bintik-bintik kecil (lentisel) "
            "di permukaannya."
        ),
        "ciri": (
            "Bentuk bulat hingga agak lonjong, dengan lekukan (cekungan) yang cukup jelas di "
            "bagian atas tempat tangkai menempel dan cekungan serupa (kadang lebih dangkal) di "
            "bagian bawah. Kulitnya cenderung mulus, mengilap, dan tipis, tanpa pori-pori besar "
            "yang mencolok. Saat dipegang terasa cukup padat dan berat untuk ukurannya, dan "
            "permukaannya bisa terasa sedikit berlilin akibat lapisan alami buah."
        ),
        "tier": "apple",
    },
    "Orange": {
        "nama": "Orange (Jeruk)",
        "warna_khas": (
            "Oranye pekat dan relatif merata di seluruh permukaan, kadang dengan semburat "
            "kuning di bagian tertentu tergantung tingkat kematangan. Warnanya cenderung lebih "
            "seragam dibanding apel, tanpa gradasi warna yang mencolok antar sisi buah."
        ),
        "ciri": (
            "Bentuk bulat hampir sempurna dengan bagian atas-bawah yang lebih rata/tidak terlalu "
            "berlekuk dibanding apel. Ciri paling khasnya ada di teksturnya: kulit berpori "
            "(dimpled), sedikit kasar saat diraba, dan mengandung banyak kelenjar minyak kecil "
            "yang membuat permukaannya terlihat bertekstur, bukan mengilap licin seperti apel. "
            "Kulitnya juga relatif lebih tebal dan sedikit lebih empuk saat ditekan dibanding "
            "kulit apel yang keras."
        ),
        "tier": "orange",
    },
}

MODEL_INFO = {
    "accuracy": 0.9313,
    "f1": 0.9308,
}

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

    .stApp {{ background: {t['bg']} !important; overflow-x: hidden !important; }}
    html, body {{ overflow-x: hidden !important; }}
    .block-container {{ 
        padding-top: 2rem !important; 
        padding-bottom: 3rem !important;
        max-width: 680px !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important; 
    }}

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
    .result-conf-track {{
        margin: 0.75rem auto 0 auto; max-width: 320px; height: 8px; border-radius: 6px;
        background: rgba(255,255,255,0.32); overflow: hidden;
    }}
    .result-conf-fill {{ height: 100%; border-radius: 6px; background: #FFFFFF; }}

    /* Box Karakteristik Visual & Interpretasi: Judul dan isi kartu rapi di dalam */
    .visual-outer-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .visual-main-header {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {t['text']} !important;
        margin: 0 0 1.1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .visual-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    .visual-item-card {{
        background: {t['bg']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
    }}
    .visual-item-full {{
        background: {t['bg']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0;
    }}
    .visual-header {{
        font-weight: 800;
        font-size: 0.88rem;
        letter-spacing: 0.5px;
        color: {t['primary']} !important;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }}
    .visual-tags {{
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 0.45rem;
        color: {t['text']} !important;
    }}
    .visual-item-card p, .visual-item-full p {{
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.55;
        color: {t['text']} !important;
    }}
    @media (max-width: 580px) {{
        .visual-grid {{ grid-template-columns: 1fr; }}
    }}

    /* Grid 2 kolom untuk kartu "Konfigurasi Model" di halaman Tentang */
    .config-grid {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 2rem;
    }}
    .config-item .config-label {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
        color: {t['muted']} !important; margin-bottom: 0.2rem;
    }}
    .config-item .config-value {{ font-size: 0.95rem; font-weight: 700; color: {t['text']} !important; }}
    @media (max-width: 420px) {{
        .config-grid {{ grid-template-columns: 1fr; }}
    }}
    .note-block {{ padding: 0.2rem 0.3rem 1rem 0.3rem; }}
    .note-block p {{ color: {t['muted']} !important; font-size: 0.85rem; line-height: 1.55; }}
    .note-block b {{ color: {t['muted']} !important; font-size: 0.85rem; letter-spacing: 0.4px; }}

    .barrow {{ display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }}
    .barrow-label {{ width: 100px; font-size: 0.8rem; color: {t['text']} !important; flex-shrink: 0; }}
    .barrow-track {{ flex: 1; background: {t['track']}; border-radius: 6px; height: 9px; overflow: hidden; }}
    .barrow-fill {{ height: 100%; border-radius: 6px; }}
    .barrow-pct {{ width: 44px; text-align: right; font-size: 0.78rem; color: {t['text']} !important; }}

    /* =======================================================================
       KOMPONEN FILE UPLOADER MODERN: SINGLE CLICKABLE DRAG & DROP BOX
       ======================================================================= */
    [data-testid="stFileUploader"] {{
        width: 100% !important;
        margin-bottom: 1.2rem !important;
    }}
    [data-testid="stFileUploader"] > div {{
        padding: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        position: relative !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1.5px dashed #A7D7C5 !important;
        border-radius: 11px !important;
        padding: 16px 22px !important;
        min-height: 76px !important;
        height: auto !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 16px !important;
        cursor: pointer !important;
        box-shadow: none !important;
        transition: background 180ms ease, border-color 180ms ease !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        text-align: left !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        background: #F4FBF7 !important;
        background-color: #F4FBF7 !important;
        border-color: #52B788 !important;
        border-style: dashed !important;
    }}

    /* 1. UPLOAD CLOUD ICON (Lucide CloudUpload SVG, 24px) */
    [data-testid="stFileUploaderDropzone"]::before {{
        content: "" !important;
        display: block !important;
        width: 24px !important;
        height: 24px !important;
        min-width: 24px !important;
        max-width: 24px !important;
        flex-shrink: 0 !important;
        background-color: #2D6A4F !important;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E") no-repeat center / contain !important;
        z-index: 2 !important;
        pointer-events: none !important;
        margin: 0 !important;
    }}

    /* Container Blok Teks (Rata Kiri, Vertikal Bertingkat) */
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: center !important;
        text-align: left !important;
        gap: 3px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        flex: 1 1 auto !important;
    }}

    /* 2. PRIMARY TEXT: "Drag and drop file here" (Baris Atas) */
    [data-testid="stFileUploaderDropzoneInstructions"]::before {{
        content: "Drag and drop file here" !important;
        display: block !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #2D3748 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.1px !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: left !important;
        visibility: visible !important;
    }}

    /* 3. SECONDARY TEXT: "Limit 200MB per file • JPG, JPEG, PNG" (Tepat di bawah baris atas, titik horizontal sama persis) */
    [data-testid="stFileUploaderDropzoneInstructions"]::after {{
        content: "Limit 200MB per file • JPG, JPEG, PNG" !important;
        display: block !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        color: #718096 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: left !important;
        visibility: visible !important;
    }}

    /* Bersihkan teks/node bawaan agar hanya teks eksplisit kita yang muncul */
    [data-testid="stFileUploaderDropzoneInstructions"] > * {{
        display: none !important;
    }}

    /* Sembunyikan elemen/svg/section bawaan Streamlit di dalam dropzone */
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] section {{
        display: none !important;
    }}

    /* Buat seluruh area dropzone klik-able untuk membuka file picker */
    [data-testid="stFileUploaderDropzone"] button {{
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 10 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }}

    /* Chip file yang sudah diunggah & semua elemen di dalamnya */
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploader"] ul,
    [data-testid="stFileUploader"] li,
    [data-testid="stFileUploader"] [data-testid*="file"],
    [data-testid="stFileUploader"] [data-testid*="File"] {{
        background-color: {t['card']} !important;
        background: {t['card']} !important;
        border-radius: 10px !important;
        border: 1px solid {t['border']} !important;
        color: {t['text']} !important;
        padding: 0.4rem 0.8rem !important;
    }}
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploader"] ul *,
    [data-testid="stFileUploader"] li * {{
        background-color: transparent !important;
        color: {t['text']} !important;
        fill: {t['text']} !important;
    }}
    /* Tombol X / Delete pada file uploader */
    [data-testid="stFileUploaderDeleteBtn"],
    [data-testid="stFileUploaderFileData"] button,
    [data-testid="stFileUploaderFile"] button,
    [data-testid="stFileUploader"] button[kind="secondary"],
    [data-testid="stFileUploader"] button[aria-label*="delete" i],
    [data-testid="stFileUploader"] button[aria-label*="remove" i],
    [data-testid="stFileUploader"] button[aria-label*="close" i] {{
        background-color: {t['primary']} !important;
        border: none !important;
        border-radius: 50% !important;
        color: {t['primary_text']} !important;
        padding: 0.2rem !important;
        width: 24px !important;
        height: 24px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }}
    [data-testid="stFileUploaderDeleteBtn"] svg,
    [data-testid="stFileUploaderDeleteBtn"] *,
    [data-testid="stFileUploaderFileData"] button svg,
    [data-testid="stFileUploaderFile"] button svg,
    [data-testid="stFileUploader"] button[aria-label*="delete" i] svg {{
        color: {t['primary_text']} !important;
        fill: {t['primary_text']} !important;
        transform: rotate(45deg);
    }}

    /* Perbaiki tombol overlay fullscreen/zoom di pojok gambar agar tidak hitam pekat polos */
    [data-testid="stImage"] button,
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    [data-testid="StyledFullScreenButton"] {{
        background: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
        color: {t['primary']} !important;
        opacity: 0.85 !important;
    }}
    [data-testid="stImage"] button svg,
    button[title="View fullscreen"] svg,
    [data-testid="StyledFullScreenButton"] svg {{
        fill: {t['primary']} !important;
        color: {t['primary']} !important;
    }}
    [data-testid="stImage"] {{
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        margin: 0 auto !important;
    }}
    [data-testid="stImage"] img {{
        margin: 0 auto !important;
        display: block !important;
        border-radius: 12px !important;
        border: 1px solid {t['border']} !important;
    }}
    [data-testid="stImageCaption"] {{
        text-align: center !important;
        color: {t['muted']} !important;
        font-size: 0.82rem !important;
        margin-top: 0.35rem !important;
    }}

    /* Sidebar navigasi */
    section[data-testid="stSidebar"] {{ background: {t['card']} !important; border-right: 1px solid {t['border']}; }}
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid*="Sidebar"][data-testid*="ollaps"],
    [aria-label*="sidebar" i],
    [aria-label*="Sidebar" i] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="stSidebarCollapseButton"] svg,
    [data-testid*="Sidebar"][data-testid*="ollaps"] svg,
    [aria-label*="sidebar" i] svg,
    [aria-label*="Sidebar" i] svg {{
        fill: {t['primary']} !important;
        color: {t['primary']} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div.sidebar-brand {{ color: {t['text']} !important; }}
    .sidebar-brand {{ font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 800; padding: 0.3rem 0 1rem 0; text-align: center; }}
    section[data-testid="stSidebar"] hr {{ border-color: {t['border']} !important; border-top: 1px solid {t['border']} !important; opacity: 1 !important; margin: 1rem 0 !important; }}

    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    section[data-testid="stSidebar"] [data-testid="element-container"],
    section[data-testid="stSidebar"] [data-testid="stButton"] {{
        width: 100% !important;
    }}

    section[data-testid="stSidebar"] .stButton button {{
        display: flex !important;
        background: transparent !important; color: {t['text']} !important; border: none !important;
        text-align: left !important; justify-content: flex-start !important; align-items: center !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.7rem !important; border-radius: 8px !important; box-shadow: none !important;
        width: 100% !important; box-sizing: border-box !important; margin: 0 !important;
    }}
    section[data-testid="stSidebar"] .stButton button * {{
        color: {t['text']} !important; font-weight: 500 !important;
        text-align: left !important; justify-content: flex-start !important;
        width: 100% !important; display: flex !important;
    }}

    section[data-testid="stSidebar"] .stButton button:hover {{ background: {t['track']} !important; }}
    section[data-testid="stSidebar"] .stButton button:hover p,
    section[data-testid="stSidebar"] .stButton button:hover div,
    section[data-testid="stSidebar"] .stButton button:hover span {{ color: {t['text']} !important; }}

    section[data-testid="stSidebar"] button[kind="primary"] {{ background: {t['primary_dark']} !important; justify-content: flex-start !important; }}
    section[data-testid="stSidebar"] button[kind="primary"] * {{
        color: {t['primary_text']} !important; font-weight: 700 !important;
        text-align: left !important; justify-content: flex-start !important;
        width: 100% !important; display: flex !important;
    }}

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


VISUAL_INFO = {
    "Apple": {
        "bentuk_tags": "Bulat • Agak lonjong",
        "bentuk_desc": (
            "Proporsi dan kontur buah memberikan bentuk yang khas, terutama pada bagian atas "
            "dan bawah buah."
        ),
        "tekstur_tags": "Halus • Mengilap • Sedikit berlilin",
        "tekstur_desc": (
            "Kondisi permukaan kulit dapat membantu membedakan karakter visual buah, terutama "
            "dari pantulan cahaya dan tampilan kulit pada gambar."
        ),
        "warna_tags": "Merah • Hijau • Kuning",
        "warna_desc": "Perbedaan warna dapat muncul karena varietas serta tingkat kematangan buah yang berbeda.",
    },
    "Orange": {
        "bentuk_tags": "Bulat • Rata di kedua ujung",
        "bentuk_desc": (
            "Proporsi dan kontur buah menunjukkan bentuk yang cenderung simetris, dengan bagian "
            "atas dan bawah yang lebih rata dibanding apel."
        ),
        "tekstur_tags": "Berpori • Sedikit kasar • Kulit lebih tebal",
        "tekstur_desc": (
            "Kondisi permukaan kulit membantu membedakan karakter visual buah, terutama dari "
            "tekstur pori dan pantulan cahaya yang lebih redup dibanding apel."
        ),
        "warna_tags": "Oranye • Kuning kecokelatan",
        "warna_desc": "Perbedaan warna dapat muncul karena tingkat kematangan buah, meski jeruk umumnya lebih merata dibanding apel.",
    },
}


def build_visual_texts(label: str, color_info: dict) -> dict:
    v = VISUAL_INFO[label]
    warna_desc = (
        f"Warna dominan pada gambar terdeteksi {color_info['label']} ({color_info['desc']}). "
        f"{v['warna_desc']}"
    )
    return {
        "warna_tags": v["warna_tags"],
        "warna_desc": warna_desc,
        "bentuk_tags": v["bentuk_tags"],
        "bentuk_desc": v["bentuk_desc"],
        "tekstur_tags": v["tekstur_tags"],
        "tekstur_desc": v["tekstur_desc"],
    }


def build_interpretation_text(label: str, confidence: float) -> str:
    if confidence >= 0.90:
        return (
            f"Tingkat keyakinan yang sangat tinggi ini menunjukkan kombinasi warna, bentuk, dan tekstur "
            f"pada gambar sangat konsisten dengan karakteristik kelas {label}. Hasil klasifikasi pada "
            f"tingkat ini dapat dianggap cukup andal untuk digunakan sebagai acuan."
        )
    elif confidence >= LOW_CONFIDENCE_THRESHOLD:
        return (
            f"Tingkat keyakinan ini tergolong cukup tinggi, meskipun sebagian ciri visual pada gambar "
            f"mungkin tidak sepenuhnya khas untuk kelas {label}. Pemeriksaan ulang secara manual tetap "
            f"disarankan apabila hasil ini digunakan untuk keputusan yang penting."
        )
    else:
        return (
            f"Tingkat keyakinan yang tergolong rendah ini menunjukkan ciri visual pada gambar kurang "
            f"sesuai secara meyakinkan dengan salah satu kelas. Hasil klasifikasi pada tingkat ini "
            f"sebaiknya tidak dijadikan acuan utama."
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
            <div style="font-size:1.9rem;">🍎🍊</div>
            <div class="hero-title">FruitID</div>
            <div class="hero-tagline">Apple & Orange Recognition</div>
            <div class="hero-sub">Unggah foto buah Anda untuk mengetahui apakah buah tersebut termasuk apel atau jeruk,
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
                <ol class="tips-list">
                    <li>Klik kotak di atas, atau tarik dan lepas foto buah.</li>
                    <li>Gunakan foto <b>close-up satu buah</b> dengan pencahayaan cukup dan latar polos.</li>
                    <li>Konfirmasi gambar sebelum sistem melakukan klasifikasi dan menampilkan rekomendasi penanganan.</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded)

            # Preview lebih ringkas dan rata tengah
            col_l, col_mid, col_r = st.columns([1, 1.4, 1])
            with col_mid:
                st.image(image, caption="Preview Foto (Konfirmasi)", width=200)

            st.markdown(
                f"""
                <div class="card" style="text-align:center; padding: 1.1rem 1.4rem; margin: 0.8rem 0 1rem 0;">
                    <div style="font-size:1.05rem; font-weight:800; margin-bottom:0.45rem; color:{t['text']} !important;">Konfirmasi Foto</div>
                    <div style="font-size:0.9rem; line-height:1.6; color:{t['muted']} !important;">
                        Periksa foto sebelum memulai diagnosis.<br>
                        Pastikan daun terlihat jelas dan fokus. Jika ingin mengganti foto, klik tombol × di bagian atas.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Mulai Klasifikasi →", type="primary", use_container_width=True):
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
                <div class="result-conf-track"><div class="result-conf-fill" style="width:{confidence*100:.1f}%;"></div></div>
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

        color_info = st.session_state.color_info
        visual = build_visual_texts(top_label, color_info)

        # Karakteristik Visual dibungkus di dalam 1 kotak kartu utama, judul di dalam kotak
        st.markdown(
            f"""
            <div class="visual-outer-card">
                <div class="visual-main-header">§ Karakteristik Visual</div>
                <div class="visual-grid">
                    <div class="visual-item-card">
                        <div class="visual-header">&#9670; WARNA</div>
                        <div class="visual-tags">{visual['warna_tags']}</div>
                        <p>{visual['warna_desc']}</p>
                    </div>
                    <div class="visual-item-card">
                        <div class="visual-header">&#9632; BENTUK</div>
                        <div class="visual-tags">{visual['bentuk_tags']}</div>
                        <p>{visual['bentuk_desc']}</p>
                    </div>
                </div>
                <div class="visual-item-full">
                    <div class="visual-header">&#9650; TEKSTUR PERMUKAAN</div>
                    <div class="visual-tags">{visual['tekstur_tags']}</div>
                    <p>{visual['tekstur_desc']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        interpretation = build_interpretation_text(top_label, confidence)
        st.markdown(
            f"""
            <div class="card">
                <b>&sect; Interpretasi</b>
                <p style="margin:0.6rem 0 0;">{interpretation}</p>
            </div>
            <div class="note-block">
                <b>&sect; Catatan</b>
                <p style="margin:0.4rem 0 0;">
                Nilai confidence yang ditampilkan merupakan keluaran probabilitas dari lapisan akhir
                model, bukan ukuran probabilitas sebenarnya bahwa objek pada gambar merupakan buah
                yang dimaksud. Nilai ini sebaiknya dipahami sebagai indikator relatif tingkat kepastian
                model terhadap prediksinya, bukan sebagai jaminan kebenaran hasil klasifikasi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        bars_html = "".join(bars)
        st.markdown(
            f"""
            <div class="card">
                <b>Distribusi Tingkat Keyakinan per Kelas</b>
                <div style="margin-top:0.7rem;">{bars_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("↻ Unggah Foto Lain", use_container_width=True):
            st.session_state.stage = 1
            st.session_state.image_bytes = None
            st.session_state.probs = None
            st.rerun()

    st.caption("FruitID · Model: CNN Custom")


# ----------------------------------------------------------------------------
# HALAMAN: TENTANG APLIKASI
# ----------------------------------------------------------------------------
def render_about():
    st.markdown('<div class="hero-title" style="text-align:left; font-size:1.6rem; margin-bottom:1rem;">Tentang Aplikasi</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <b>Tentang FruitID</b>
            <p style="margin:0.5rem 0 0.4rem 0;">FruitID merupakan aplikasi untuk membantu
            mengidentifikasi apakah sebuah foto buah adalah apel atau jeruk.</p>
            <p style="margin:0;">Aplikasi ini memberikan alasan prediksi berbasis analisis warna
            dominan gambar, agar hasil klasifikasi lebih mudah dipahami.</p>
        </div>

        <div class="card">
            <b>Konfigurasi Model</b>
            <p style="margin:0.5rem 0 0.9rem 0; font-weight:600;">Custom CNN (dibangun dari nol)</p>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">Arsitektur</div>
                    <div class="config-value">CNN VGG-style, 3 blok konvolusi</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Klasifikasi</div>
                    <div class="config-value">Biner, 2 kelas</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Input</div>
                    <div class="config-value">128 × 128 RGB</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Optimizer</div>
                    <div class="config-value">Adam (lr 1e-4)</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Output</div>
                    <div class="config-value">Apple / Orange</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Pembelajaran</div>
                    <div class="config-value">Dilatih dari nol (scratch)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, m1, m2, _ = st.columns([1, 2, 2, 1])
    m1.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
    m2.metric("F1 Score", f"{MODEL_INFO['f1']:.4f}")

    st.markdown(
        """
        <div class="card">
            <b>Informasi Aplikasi</b>
            <p style="margin:0.6rem 0 0 0;">FruitID<br>Model: CNN Custom</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR — NAVIGASI UTAMA
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">FruitID</div>', unsafe_allow_html=True)

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
