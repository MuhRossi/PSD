# Commented out IPython magic to ensure Python compatibility.
# %pip install tabulate numpy pandas matplotlib seaborn scikit-learn scipy -q

"""## Library yang Digunakan"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.special import expit as sigmoid
from scipy.special import expit

"""## Mengambil data dari heart.csv"""

#  Baca file CSV
df = pd.read_csv('heart.csv')

#  Tampilkan seluruh data dalam format tabel rapi
print("===== Tabel Data Penyakit Jantung =====")
df.head()

"""## Preprocessing Data

### Cek informasi data
"""

# Tampilkan jumlah data awal
print(f"Jumlah data awal: {len(df)}")

# Hapus data yang memiliki missing value (NaN) saja
df_cleaned = df.dropna()
print(f"\nJumlah data setelah hapus missing value: {len(df_cleaned)}")

# Tampilkan informasi struktur dataset
print("===== Informasi Dataset =====")
df_cleaned.info()

# Tampilkan statistik deskriptif
print("\n===== Statistik Deskriptif =====")
print(df_cleaned.describe())

# Visualisasi distribusi kolom numerik
print("\n===== Visualisasi Distribusi Kolom Numerik =====")
numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Tentukan jumlah subplot per baris & baris total
n_cols = 3
n_rows = -(-len(numeric_columns) // n_cols)  # ceil division

plt.figure(figsize=(16, 4 * n_rows))

# Loop visualisasi histogram per kolom
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df_cleaned[col], kde=True, color='skyblue')
    plt.title(f'Distribusi: {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()

"""### distribusi kelas target"""

# Cek apakah kolom 'target' ada
if 'target' in df_cleaned.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_cleaned, x='target', palette='Set2')

    # Judul dan label
    plt.title('Distribusi Kelas Target (Penyakit Jantung)')
    plt.xlabel('Target (0 = Tidak, 1 = Ya)')
    plt.ylabel('Jumlah')

    # Label sumbu x
    plt.xticks([0, 1], ['Tidak Ada Penyakit', 'Ada Penyakit'])

    # Tambahkan grid untuk memperjelas
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Tata letak agar tidak terpotong
    plt.tight_layout()
    plt.show()
else:
    print("Kolom 'target' tidak ditemukan di df_cleaned.")

"""## Normalisasi data menggunakan Z-Score

Z-Score (atau standar skor) adalah metode untuk menormalisasi data numerik agar memiliki:

* Rata-rata (mean) = 0
* Standar deviasi (std) = 1


## Rumus Z-Score

$$
Z = \frac{X - \mu}{\sigma}
$$

Keterangan:

* $X$: Nilai asli
* $\mu$: Rata-rata (mean) dari kolom tersebut
* $\sigma$: Standar deviasi dari kolom tersebut
* $Z$: Nilai yang sudah dinormalisasi (Z-Score)

---

## Contoh Perhitungan Manual

Misalkan kita punya data kolom `age` dari 5 pasien:

| No | Age |
| -- | --- |
| 1  | 63  |
| 2  | 59  |
| 3  | 55  |
| 4  | 54  |
| 5  | 65  |

### 1. Hitung Rata-rata (Mean)

$$
\mu = \frac{63 + 59 + 55 + 54 + 65}{5} = \frac{296}{5} = 59.2
$$

### 2. Hitung Standar Deviasi (σ)

$$
\sigma = \sqrt{\frac{(63 - 59.2)^2 + (59 - 59.2)^2 + \dots + (65 - 59.2)^2}{5}} = \sqrt{\frac{88.8}{5}} \approx 4.21
$$

### 3. Hitung Z-Score Masing-masing Nilai

Contoh untuk data pertama (`age = 63`):

$$
Z = \frac{63 - 59.2}{4.21} = \frac{3.8}{4.21} \approx 0.90
$$

Data kedua (`age = 59`):

$$
Z = \frac{59 - 59.2}{4.21} = \frac{-0.2}{4.21} \approx -0.05
$$



## Tujuan Penggunaan Z-Score:

* Menyamakan skala semua fitur (penting untuk algoritma seperti ELM, SVM, KNN, dll)
* Menghindari bias terhadap fitur dengan rentang nilai besar
"""

# ============================================================
# 1. Pisahkan fitur (X) dan label (y)
# ============================================================
X = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']

# ============================================================
# 2. Inisialisasi StandardScaler
# ============================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# ============================================================
# 3. Lakukan normalisasi Z-Score
# ============================================================
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. Buat ulang DataFrame hasil normalisasi
# ============================================================
import pandas as pd
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("===== 5 Data Pertama Setelah Normalisasi Z-Score =====")
print(X_scaled_df.head())

# ============================================================
# 5. Visualisasi Sebelum dan Sesudah Normalisasi
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=X, color='lightgray')
plt.title("Sebelum Normalisasi (Z-Score)")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=X_scaled_df, color='skyblue')
plt.title("Sesudah Normalisasi (Z-Score)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""### Pembagian Data"""

# ============================================================
# 1. Gunakan data yang sudah dinormalisasi sebagai fitur
# ============================================================
X = X_scaled_df        # fitur (sudah dinormalisasi)
y = df_cleaned['target']  # label

# ============================================================
# 2. Bagi data menjadi 70% training dan 30% testing
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # menjaga proporsi kelas tetap seimbang
)

# ============================================================
# 3. Tampilkan ringkasan hasil pembagian
# ============================================================
print("===== Pembagian Data =====")
print(f"Jumlah data total   : {len(X)}")
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing : {len(X_test)}")

"""### MODELING ELM"""

import numpy as np, random, os
seed = 42
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# ============================================================
# 1. Pembagian Data 70% Training dan 30% Testing
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=seed
)

# ============================================================
# 2. One-Hot Encoding untuk label
# ============================================================
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_test_oh = encoder.transform(y_test.to_numpy().reshape(-1, 1))

# ============================================================
# 3. Fungsi aktivasi sigmoid
# ============================================================
import numpy as np

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# ============================================================
# 4. Kelas ELM
# ============================================================
class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation=sigmoid_activation, random_state=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.random_state = random_state

        if self.random_state is not None:
           np.random.seed(self.random_state)

        self.input_weights = np.random.randn(self.input_size, self.hidden_size)
        self.bias = np.random.randn(self.hidden_size)

    def fit(self, X, y):
        H = self.activation(np.dot(X, self.input_weights) + self.bias)
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self.activation(np.dot(X, self.input_weights) + self.bias)
        output = np.dot(H, self.output_weights)
        return np.argmax(output, axis=1)

# ============================================================
# 5. Inisialisasi dan Latih Model ELM
# ============================================================
elm = ELMClassifier(input_size=X_train.shape[1], hidden_size=150, random_state=seed)
elm.fit(X_train, y_train_oh)

# ============================================================
# 6. Prediksi pada Data Testing
# ============================================================
y_pred = elm.predict(X_test)

# ============================================================
# 7. Evaluasi Hasil
# ============================================================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n===== Hasil Evaluasi Model ELM =====")
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\n===== Confusion Matrix =====")
print(cm)

print("\nKlasifikasi Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 8. Visualisasi Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.categories_[0],
            yticklabels=encoder.categories_[0])
plt.title("Confusion Matrix - ELM")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()

# ============================================================
# MODELING SVM
# ============================================================
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Inisialisasi model SVM
svm_model = SVC(kernel='rbf', random_state=seed)

# 2. Latih model
svm_model.fit(X_train, y_train.values.ravel())

# 3. Prediksi data testing
y_pred_svm = svm_model.predict(X_test)

# 4. Evaluasi hasil
print("\n===== Hasil Evaluasi Model SVM =====")
print(f"Akurasi: {accuracy_score(y_test, y_pred_svm):.4f}")

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\n===== Confusion Matrix =====")
print(cm_svm)

print("\nKlasifikasi Report:\n", classification_report(y_test, y_pred_svm))

# 5. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()

# ============================================================
# MODELING RANDOM FOREST
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Inisialisasi model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)

# 2. Latih model
rf_model.fit(X_train, y_train.values.ravel())

# 3. Prediksi data testing
y_pred_rf = rf_model.predict(X_test)

# 4. Evaluasi hasil
print("\n===== Hasil Evaluasi Model Random Forest =====")
print(f"Akurasi: {accuracy_score(y_test, y_pred_rf):.4f}")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\n===== Confusion Matrix =====")
print(cm_rf)

print("\nKlasifikasi Report:\n", classification_report(y_test, y_pred_rf))

# 5. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()

import joblib
from sklearn.preprocessing import StandardScaler

# Misalnya kamu pakai ini sebelumnya untuk normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, "scaler_elm_heart.joblib")
print("Scaler berhasil disimpan!")

import joblib


# Simpan model dan encoder
joblib.dump(elm, 'model_elm_heart.joblib')
joblib.dump(encoder, 'encoder_elm_heart.joblib')
print("✅ Model ELM berhasil disimpan!")

"""HASIL WEB

https://kkmmfkxy66fu2hwjfxlsmn.streamlit.app/

contoh input data tidak beresiko
| Fitur                                     | Nilai     |
| ----------------------------------------- | --------- |
| **Umur**                                  | 35        |
| **Jenis Kelamin**                         | Perempuan |
| **Tipe Nyeri Dada (0–3)**                 | 0         |
| **Tekanan Darah (mmHg)**                  | 120       |
| **Kolesterol (mg/dl)**                    | 180       |
| **Gula Darah > 120 mg/dl (1=Ya,0=Tidak)** | 0         |
| **Hasil ECG (0–2)**                       | 0         |
| **Denyut Jantung Maksimum**               | 170       |
| **Angina karena olahraga (1=Ya,0=Tidak)** | 0         |
| **ST Depression**                         | 0.0       |
| **Slope ST (0–2)**                        | 1         |
| **Jumlah pembuluh besar (0–4)**           | 0         |
| **Thalassemia (0–3)**                     | 2         |

contoh beresiko
| Fitur                                     | Nilai     |
| ----------------------------------------- | --------- |
| **Umur**                                  | 63        |
| **Jenis Kelamin**                         | Laki-laki |
| **Tipe Nyeri Dada (0–3)**                 | 3         |
| **Tekanan Darah (mmHg)**                  | 150       |
| **Kolesterol (mg/dl)**                    | 260       |
| **Gula Darah > 120 mg/dl (1=Ya,0=Tidak)** | 1         |
| **Hasil ECG (0–2)**                       | 2         |
| **Denyut Jantung Maksimum**               | 120       |
| **Angina karena olahraga (1=Ya,0=Tidak)** | 1         |
| **ST Depression**                         | 3.5       |
| **Slope ST (0–2)**                        | 0         |
| **Jumlah pembuluh besar (0–4)**           | 2         |
| **Thalassemia (0–3)**                     | 1         |
"""
