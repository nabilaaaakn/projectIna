import streamlit as st
import numpy as np
import pickle

# Load model yang sudah dilatih
model_filename = "ML_Jantung.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Judul aplikasi
st.title("🔍 Klasifikasi Penyakit Jantung")

# Deskripsi singkat
st.markdown("""
💖 **Aplikasi ini menggunakan Machine Learning untuk mengklasifikasi apakah seseorang memiliki penyakit jantung atau tidak.**  
Masukkan data pasien di bawah ini, lalu klik tombol **Prediksi** untuk melihat hasilnya.
""")

# Membuat form input dengan kolom yang sesuai
st.sidebar.header("🔢 Masukkan Data Pasien")

age = st.sidebar.number_input("Usia", min_value=1, max_value=120, value=55)
sex = st.sidebar.selectbox("Jenis Kelamin", [("Laki-laki", 1), ("Perempuan", 0)])
cp = st.sidebar.selectbox("Tipe Nyeri Dada", [("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)])
trestbps = st.sidebar.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=200, value=120)
thalach = st.sidebar.number_input("Detak Jantung Maksimum", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Nyeri Dada saat Olahraga?", [("Tidak", 0), ("Ya", 1)])
oldpeak = st.sidebar.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope Segmen ST", [("Up", 2), ("Flat", 1), ("Down", 0)])
ca = st.sidebar.number_input("Jumlah Pembuluh Tersumbat", min_value=0, max_value=3, value=1)
thal = st.sidebar.selectbox("Hasil Tes Thalium", [("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)])

# Mengubah pilihan menjadi format numerik
sex = sex[1]
cp = cp[1]
exang = exang[1]
slope = slope[1]
thal = thal[1]

# Membuat array input
patient_data = np.array([[age, sex, cp, trestbps, thalach, exang, oldpeak, slope, ca, thal]])

# Tombol prediksi
if st.sidebar.button("🔮 Prediksi"):
    prediction = model.predict(patient_data)

    # Menampilkan hasil prediksi
    st.subheader("🩺 Hasil Prediksi")
    
    if prediction[0] == 1:
        st.error("🚨 **Pasien berisiko memiliki penyakit jantung!**")
        st.markdown("🔴 **Segera konsultasikan ke dokter untuk pemeriksaan lebih lanjut.**")
    else:
        st.success("✅ **Pasien tidak memiliki penyakit jantung.**")
        st.markdown("🟢 **Tetap jaga pola hidup sehat!**")

# Footer
st.markdown("---")
st.markdown("📌 *Dibuat oleh Nabila dan Ichsan❤️*")