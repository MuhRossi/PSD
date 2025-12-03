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
