# Apple vs Orange Classifier — Deployment

## Isi folder
- `app.py` — aplikasi Streamlit (UI prediksi + perbandingan model + versioning)
- `requirements.txt` — dependency untuk Streamlit Cloud / lokal
- `custom_cnn_model.h5` — **(perlu ditambahkan)** hasil export dari notebook Week 2–4
- `pretrained_mobilenetv2_model.h5` — **(perlu ditambahkan)** hasil export dari notebook Week 2–4

Notebook kamu (cell "7. Menyimpan Model") sudah menyimpan kedua file `.h5` ini —
tinggal salin dari hasil run Kaggle/Colab ke folder yang sama dengan `app.py`.

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push folder ini (`app.py`, `requirements.txt`, kedua file `.h5`) ke repo GitHub.
2. Buka https://share.streamlit.io → **New app** → pilih repo & branch → `app.py` sebagai entry point.
3. Deploy, lalu salin tautan aplikasi untuk dikumpulkan.

## Catatan ukuran model
`pretrained_mobilenetv2_model.h5` cukup besar karena menyertakan seluruh bobot backbone
MobileNetV2. Jika repo GitHub menolak file besar, gunakan Git LFS atau host file model
di Google Drive/HuggingFace lalu unduh otomatis di awal `app.py` (opsional, bisa
ditambahkan sebagai poin optimasi lanjutan).
