
import streamlit as st
import pandas as pd, numpy as np
import pickle, os
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(page_title="My Portfolio with Streamlit", layout="wide")

st.title("My Portfolio with Streamlit")
st.markdown("Portofolio dan aplikasi prediksi nilai UTBK per subtes untuk berbagai jurusan/prodi.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Upload & Predict","Visualisasi Data","Tentang Saya"])

model_path = os.path.join(BASE_DIR, "model_pipeline.pkl")

@st.cache_resource
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)

if page == "Home":
    st.header("Selamat datang!")
    st.write("Aplikasi ini dibuat sebagai portofolio untuk menampilkan analisis dan prediksi nilai UTBK.")

elif page == "Upload & Predict":
    st.header("Upload data peserta (CSV) untuk prediksi per subtes")
    uploaded = st.file_uploader("Upload CSV (format kolom: TO 1, TO 2, ..., JURUSAN/PRODI, ...)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview input:")
        st.dataframe(df.head())
        if st.button("Run Prediction"):
            model = load_model()
            preds = model.predict(df)
            preds_df = pd.DataFrame(preds, columns=["PU","PK","PPU","PBM","LIND","LING"])
            st.success("Prediksi selesai")
            st.dataframe(preds_df.head())
            st.download_button("Download predictions CSV", preds_df.to_csv(index=False), "predictions.csv", "text/csv")

elif page == "Visualisasi Data":
    st.header("Visualisasi dataset nilai UTBK")
    csvp = os.path.join(BASE_DIR, "nilai_utbk_cleaned.csv")
    if os.path.exists(csvp):
        d = pd.read_csv(csvp)
        st.write("Preview cleaned data:")
        st.dataframe(d.head())
        st.subheader("Distribusi salah satu subtes (PU)")
        st.bar_chart(d["PU"].fillna(0).astype(float).head(100))
    else:
        st.info("Data bersih tidak ditemukan.")

elif page == "Tentang Saya":
    st.header("Tentang Saya")
    st.markdown("""
**Rusdi Ahmad**  
Master's in Mathematics. Mathematics educator and AI/ML enthusiast.  
Email: rusdiahmad979@gmail.com
""")
