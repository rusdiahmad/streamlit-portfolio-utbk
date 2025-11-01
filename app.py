import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pickle
import os

# ---------------------------
# 🔧 KONFIGURASI DASAR
# ---------------------------
st.set_page_config(page_title="My Portfolio with Streamlit", layout="wide")

st.title("My Portfolio with Streamlit")
st.markdown("""
Aplikasi portofolio berbasis Streamlit untuk menampilkan analisis dan prediksi nilai **UTBK per subtes dan jurusan/prodi**.
""")

# ---------------------------
# 🔗 URL DATASET dari GitHub
# ---------------------------
# Ganti URL ini dengan raw link file Excel kamu di GitHub
GITHUB_RAW_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/NILAI_UTBK_ANGK_4.xlsx"

# ---------------------------
# 🧠 Fungsi Muat Model dan Data
# ---------------------------

@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    try:
        r = requests.get(GITHUB_RAW_URL)
        if r.status_code != 200:
            st.error(f"Gagal memuat data dari GitHub. Status code: {r.status_code}")
            return None
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="DATABASE")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None

# ---------------------------
# 🧭 SIDEBAR NAVIGASI
# ---------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "Beranda",
    "Visualisasi Data",
    "Prediksi Nilai",
    "Tentang Saya"
])

# ---------------------------
# 📊 BERANDA
# ---------------------------
if page == "Beranda":
    st.header("Selamat Datang di Portofolio Rusdi Ahmad 🎓")
    st.write("""
Aplikasi ini merupakan implementasi tugas *Portfolio Building with Streamlit* yang
menampilkan analisis dan prediksi **nilai UTBK per subtes** berdasarkan jurusan/prodi pilihan.
""")

# ---------------------------
# 📈 VISUALISASI DATA
# ---------------------------


if page == "Data Viz":
    st.header("Data Visualization")
    data_path = "data/NILAI_UTBK_ANGK_4.xlsx"

if os.path.exists(data_path):
    df = pd.read_excel(data_path)
    st.success("✅ Dataset UTBK berhasil dimuat otomatis dari folder data/")
    st.dataframe(df.head())
else:
    st.error("❌ Dataset belum ditemukan. Harap tambahkan file ke folder data/")


# ---------------------------
# 🤖 PREDIKSI NILAI
# ---------------------------
if page == "Prediksi Nilai":
    st.header("Prediksi Nilai UTBK per Subtes")
    df = load_data()
    if df is not None:
        model = load_model()
        feature_cols = ['TO 1','TO 2','TO 3','TO 4','TO 5','TO 6','TO 7',
                        'RATA- RATA TO 4 S.D 7','ESTIMASI RATA-RATA',
                        'Rata-rata','Ranking','RUMPUN','JURUSAN/PRODI']
        data_pred = df[feature_cols].dropna()
        preds = model.predict(data_pred)
        preds_df = pd.DataFrame(preds, columns=["PU","PK","PPU","PBM","LIND","LING"])
        st.success("Prediksi Berhasil ✅")
        st.dataframe(preds_df.head())

        csv = preds_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil Prediksi (CSV)", csv, "prediksi_utbk.csv", "text/csv")
    else:
        st.warning("Data tidak tersedia untuk prediksi.")

# ---------------------------
# 👤 TENTANG SAYA
# ---------------------------
elif page == "Tentang Saya":
    st.header("Tentang Saya")
    st.markdown("""
**Rusdi Ahmad**  
Magister Matematika • Pengajar & Peneliti AI/ML  
Email: rusdiahmad979@gmail.com  

Portofolio ini menampilkan kemampuan dalam:
- Analisis data & visualisasi dengan Python  
- Pembuatan model Machine Learning  
- Implementasi Streamlit & deployment di cloud  
""")

