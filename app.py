import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import ELM dari file terpisah
from elm_classifier import ELMClassifier, sigmoid_activation


# ============================================================
# TITLE
# ============================================================
st.title("📊 Prediksi Penyakit Jantung – ELM, SVM, Random Forest")

st.markdown("""
Aplikasi ini melakukan:
- Preprocessing & normalisasi dataset
- Visualisasi
- Penjelasan rumus Z-Score
- Pelatihan model ELM, SVM, RandomForest
- Menyimpan model `.joblib`
""")


# ============================================================
# 1. LOAD DATASET
# ============================================================
st.header("📥 1. Memuat Dataset")
df = pd.read_csv("heart.csv")
st.dataframe(df)


# ============================================================
# 2. CLEANING
# ============================================================
st.header("🧹 2. Membersihkan Missing Value")
df_clean = df.dropna()
st.write(f"Jumlah data setelah bersih: **{len(df_clean)}**")


# ============================================================
# 3. VISUALISASI DISTRIBUSI
# ============================================================
st.header("📈 3. Visualisasi Kolom Numerik")
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

fig, ax = plt.subplots(len(num_cols), 1, figsize=(8, 4 * len(num_cols)))
for i, col in enumerate(num_cols):
    sns.histplot(df_clean[col], kde=True, ax=ax[i])
    ax[i].set_title(f"Distribusi {col}")
st.pyplot(fig)


# ============================================================
# 4. Penjelasan Z-SCORE
# ============================================================
st.header("📘 4. Penjelasan Rumus Z-Score")

st.markdown("### Rumus Umum:")
st.latex(r"Z = \frac{X - \mu}{\sigma}")

st.markdown("### Contoh Perhitungan Mean:")
st.latex(r"""
\mu = \frac{63 + 59 + 55 + 54 + 65}{5}
= 59.2
""")

st.markdown("### Contoh Standar Deviasi:")
st.latex(r"""
\sigma = \sqrt{
\frac{(63-59.2)^2 + (59-59.2)^2 + (55-59.2)^2 + (54-59.2)^2 + (65-59.2)^2}{5}
}
\approx 4.21
""")

st.markdown("### Contoh Z-Score:")
st.latex(r"""
Z = \frac{63 - 59.2}{4.21} \approx 0.90
""")


# ============================================================
# 5. NORMALISASI
# ============================================================
st.header("⚙ 5. Normalisasi Z-Score")
X = df_clean.drop("target", axis=1)
y = df_clean["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write("Data setelah normalisasi:")
st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())


# ============================================================
# 6. SPLIT DATA
# ============================================================
st.header("✂ 6. Pembagian Data")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
st.write(f"Training data: **{len(X_train)}**")
st.write(f"Testing data: **{len(X_test)}**")


# ============================================================
# 7. TRAINING MODEL
# ============================================================
st.header("🤖 7. Training Model Machine Learning")


# ---------------------
# ELM
# ---------------------
st.subheader("🔵 Model ELM")

encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))

elm = ELMClassifier(
    input_size=X_train.shape[1],
    hidden_size=150,
    random_state=42
)
elm.fit(X_train, y_train_oh)
y_pred_elm = elm.predict(X_test)

st.write("Akurasi ELM:", accuracy_score(y_test, y_pred_elm))


# ---------------------
# SVM
# ---------------------
st.subheader("🟣 Model SVM")

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

st.write("Akurasi SVM:", accuracy_score(y_test, y_pred_svm))


# ---------------------
# RANDOM FOREST
# ---------------------
st.subheader("🟢 Model Random Forest")

rf_model = RandomForestClassifier(n_estimators=120, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.write("Akurasi Random Forest:", accuracy_score(y_test, y_pred_rf))


# ============================================================
# 8. SAVE MODELS
# ============================================================
st.header("💾 8. Menyimpan Model ke Folder /models/")

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler_heart.joblib")
joblib.dump(encoder, "models/encoder_heart.joblib")
joblib.dump(elm, "models/model_elm_heart.joblib")
joblib.dump(svm_model, "models/model_svm_heart.joblib")
joblib.dump(rf_model, "models/model_rf_heart.joblib")

st.success("Model berhasil disimpan ke folder /models/")


# ============================================================
# 9. FITUR PREDIKSI
# ============================================================
st.header("🩺 9. Prediksi Penyakit Jantung dari Input Manual")

st.markdown("""
Masukkan data pasien di bawah ini untuk memprediksi apakah pasien **berisiko** atau **tidak berisiko**
mengalami penyakit jantung.
""")

# ------------------------------
# FORM INPUT USER
# ------------------------------
age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
cp = st.selectbox("Tipe Nyeri Dada (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Gula Darah > 120 mg/dl", [0, 1])
restecg = st.selectbox("Hasil ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Denyut Jantung Maksimum", 60, 220, 150)
exang = st.selectbox("Angina karena olahraga", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope ST (0–2)", [0, 1, 2])
ca = st.selectbox("Jumlah pembuluh besar (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Convert gender
sex_val = 1 if sex == "Laki-laki" else 0

# ------------------------------
# Input disatukan dalam array
# ------------------------------
input_data = np.array([
    age, sex_val, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal
]).reshape(1, -1)

# ------------------------------
# Pilih model
# ------------------------------
st.subheader("Pilih Model Prediksi")
model_choice = st.selectbox(
    "Model yang digunakan:",
    ["ELM", "SVM", "Random Forest"]
)

# Load model ketika user menekan tombol
if st.button("🔮 Prediksi Sekarang"):
    
    # Normalize input
    scaler_loaded = joblib.load("models/scaler_heart.joblib")
    input_scaled = scaler_loaded.transform(input_data)

    if model_choice == "ELM":
        encoder_loaded = joblib.load("models/encoder_elm_heart.joblib")
        model = joblib.load("models/model_elm_heart.joblib")
        pred = model.predict(input_scaled)

    elif model_choice == "SVM":
        model = joblib.load("models/model_svm_heart.joblib")
        pred = model.predict(input_scaled)

    elif model_choice == "Random Forest":
        model = joblib.load("models/model_rf_heart.joblib")
        pred = model.predict(input_scaled)

    # ------------------------------
    # Hasil Prediksi
    # ------------------------------
    result = int(pred[0])

    st.subheader("🧾 Hasil Prediksi")

    if result == 0:
        st.success("✅ **Tidak Berisiko Penyakit Jantung**")
    else:
        st.error("⚠ **Berisiko Penyakit Jantung**")

    st.info(f"Model yang digunakan: **{model_choice}**")

