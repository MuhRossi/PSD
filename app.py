import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")

# ================================================================
# LOAD MODEL
# ================================================================
scaler = joblib.load("models/scaler_heart.joblib")
svm_model = joblib.load("models/model_svm_heart.joblib")
rf_model = joblib.load("models/model_rf_heart.joblib")

st.title("❤️ Prediksi Penyakit Jantung")
st.write("Masukkan data pasien untuk mendapatkan hasil prediksi.")

# ================================================================
# FORM INPUT PREDIKSI
# ================================================================
age = st.number_input("Umur", 1, 120, 35)
sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
cp = st.number_input("Tipe Nyeri Dada (0–3)", 0, 3, 0)
trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
chol = st.number_input("Kolesterol (mg/dl)", 100, 500, 180)
fbs = st.selectbox("Gula Darah > 120 mg/dl", ["Tidak", "Ya"])
restecg = st.number_input("Hasil ECG (0–2)", 0, 2, 0)
thalach = st.number_input("Denyut Jantung Maksimum", 60, 220, 170)
exang = st.selectbox("Angina karena olahraga", ["Tidak", "Ya"])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1)
slope = st.number_input("Slope ST (0–2)", 0, 2, 1)
ca = st.number_input("Jumlah pembuluh besar (0–4)", 0, 4, 0)
thal = st.number_input("Thalassemia (0–3)", 0, 3, 2)

# Convert kategori ke angka
sex = 1 if sex == "Laki-laki" else 0
fbs = 1 if fbs == "Ya" else 0
exang = 1 if exang == "Ya" else 0

# Bentuk array input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

# Normalisasi
scaled_input = scaler.transform(input_data)

# ================================================================
# PREDIKSI
# ================================================================
if st.button("🔍 Prediksi"):
    pred_svm = svm_model.predict(scaled_input)[0]
    pred_rf = rf_model.predict(scaled_input)[0]

    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🧠 SVM Model")
        st.success("Tidak Berisiko" if pred_svm == 0 else "Berisiko Tinggi")

    with col2:
        st.write("### 🌲 Random Forest")
        st.success("Tidak Berisiko" if pred_rf == 0 else "Berisiko Tinggi")
