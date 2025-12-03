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
from scipy.special import expit

# ============================================================
# TITLE
# ============================================================
st.title("📊 Prediksi Penyakit Jantung – ELM, SVM, Random Forest")

st.markdown("""
Aplikasi ini melakukan:
- Preprocessing & normalisasi dataset
- Visualisasi data
- Perhitungan Z-Score (dengan rumus LaTeX)
- Pelatihan model ELM, SVM, dan Random Forest
- Menyimpan model dalam `.joblib`
""")

# ============================================================
# LOAD DATASET
# ============================================================
st.header("📥 1. Memuat Dataset `heart.csv`")

df = pd.read_csv("heart.csv")
st.dataframe(df)

st.write(f"Jumlah data: **{len(df)}**")

# ============================================================
# CLEANING
# ============================================================
st.header("🧹 2. Membersihkan Data Missing Value")

df_clean = df.dropna()
st.write(f"Jumlah data setelah dihapus missing value: **{len(df_clean)}**")

# ============================================================
# VISUALISASI DISTRIBUSI NUMERIK
# ============================================================
st.header("📈 3. Visualisasi Distribusi Kolom Numerik")

num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

fig, ax = plt.subplots(len(num_cols), 1, figsize=(8, 4*len(num_cols)))
for i, col in enumerate(num_cols):
    sns.histplot(df_clean[col], kde=True, ax=ax[i])
    ax[i].set_title(f"Distribusi {col}")

st.pyplot(fig)

# ============================================================
# DISTRIBUSI TARGET
# ============================================================
st.header("🎯 4. Distribusi Kelas Target")

fig2, ax2 = plt.subplots()
sns.countplot(data=df_clean, x='target', palette='Set2', ax=ax2)
st.pyplot(fig2)

# ============================================================
# Z-SCORE EXPLANATION
# ============================================================
st.header("📘 5. Penjelasan Z-Score dengan Rumus")

st.markdown("""
### Rumus Z-Score
""")
st.latex(r"Z = \frac{X - \mu}{\sigma}")

st.markdown("### Contoh Perhitungan Mean:")
st.latex(r"""
\mu = \frac{63 + 59 + 55 + 54 + 65}{5}
= \frac{296}{5}
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
# NORMALISASI Z-SCORE
# ============================================================
st.header("⚙ 6. Normalisasi Z-Score")

X = df_clean.drop("target", axis=1)
y = df_clean["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write("Contoh sebelum normalisasi:")
st.dataframe(X.head())

st.write("Contoh setelah normalisasi:")
st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

# ============================================================
# SPLIT DATA
# ============================================================
st.header("✂ 7. Pembagian Data Training & Testing")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

st.write(f"Training: **{len(X_train)}** data")
st.write(f"Testing: **{len(X_test)}** data")

# ============================================================
# ELM Classifier
# ============================================================
st.header("🤖 8. Training Model ELM")

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation=sigmoid_activation, random_state=42):
        np.random.seed(random_state)
        self.input_weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)
        self.activation = activation

    def fit(self, X, y):
        H = self.activation(X @ self.input_weights + self.bias)
        self.output_weights = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self.activation(X @ self.input_weights + self.bias)
        output = H @ self.output_weights
        return np.argmax(output, axis=1)

# One-hot untuk ELM
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.values.reshape(-1, 1))

# Train ELM
elm = ELMClassifier(input_size=X_train.shape[1], hidden_size=150)
elm.fit(X_train, y_train_oh)
y_pred_elm = elm.predict(X_test)

st.write("**Akurasi ELM:**", accuracy_score(y_test, y_pred_elm))

# ============================================================
# SVM
# ============================================================
st.header("🤖 9. Training Model SVM")

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

st.write("**Akurasi SVM:**", accuracy_score(y_test, y_pred_svm))

# ============================================================
# RANDOM FOREST
# ============================================================
st.header("🌲 10. Training Model Random Forest")

rf_model = RandomForestClassifier(n_estimators=120)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.write("**Akurasi Random Forest:**", accuracy_score(y_test, y_pred_rf))

# ============================================================
# SAVE MODELS
# ============================================================
st.header("💾 11. Menyimpan Model")

joblib.dump(scaler, "scaler_heart.joblib")
joblib.dump(elm, "model_elm_heart.joblib")
joblib.dump(encoder, "encoder_elm_heart.joblib")
joblib.dump(svm_model, "model_svm_heart.joblib")
joblib.dump(rf_model, "model_rf_heart.joblib")

st.success("Semua model berhasil disimpan!")
