# Facial Skin Ageing Image Classification with CNN

Proyek klasifikasi citra untuk mendeteksi kondisi penuaan kulit wajah (*facial skin ageing*) menggunakan **Convolutional Neural Network (CNN)**, dibangun dengan TensorFlow/Keras. Model dilatih untuk mengenali beberapa kategori kondisi kulit wajah dari gambar, dan dikonversi ke berbagai format (`.h5`, `.keras`, SavedModel, TensorFlow Lite, dan TensorFlow.js) untuk kebutuhan deployment di berbagai platform (Python, mobile, dan web).

## Daftar Isi

- [Struktur Repository](#struktur-repository)
- [Fitur](#fitur)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
  - [Training Model](#training-model)
  - [Evaluasi Model](#evaluasi-model)
  - [Inference / Prediksi Gambar Baru](#inference--prediksi-gambar-baru)
  - [Konversi Model](#konversi-model)
- [Format Model yang Tersedia](#format-model-yang-tersedia)
- [Kelas Klasifikasi](#kelas-klasifikasi)
- [Requirements](#requirements)
- [Catatan Kompatibilitas](#catatan-kompatibilitas)
- [Lisensi](#lisensi)

## Struktur Repository

```
├── dataset/                          # Dataset mentah/awal
├── dataset-final/                    # Dataset yang sudah diproses/final untuk training
├── saved_model/                      # Model hasil training (.h5 / .keras / SavedModel)
├── tflite/                           # Model hasil konversi ke TensorFlow Lite
├── tfjs/                             # Model hasil konversi ke TensorFlow.js
├── code.ipynb                        # Notebook utama: preprocessing, training, evaluasi model
├── code_convert_model_to_tfjs.ipynb  # Notebook konversi model ke format TensorFlow.js
└── nenek_insidius.jpg                # Contoh gambar untuk testing inference
```

## Fitur

- Klasifikasi citra wajah untuk deteksi kondisi kulit terkait penuaan menggunakan CNN
- Evaluasi model lengkap: akurasi & loss (training/testing), confusion matrix, classification report
- Model tersedia dalam berbagai format untuk kebutuhan deployment:
  - **Keras native** (`.keras`) — untuk penggunaan di Python
  - **HDF5** (`.h5`) — kompatibilitas versi lama
  - **SavedModel** (`.pb` + `variables/`) — untuk TensorFlow Serving
  - **TensorFlow Lite** (`.tflite`) — untuk aplikasi mobile/embedded
  - **TensorFlow.js** — untuk aplikasi web/browser
- Script inference siap pakai dengan fitur pilih/upload gambar dan tampilan confidence per kelas

## Instalasi

1. Clone repository ini:
   ```bash
   git clone https://github.com/febryofibonacciamadeo/Facial_Skin_Ageing_Image_Classification_with_Convolutional_Neural_Network-CNN-.git
   cd Facial_Skin_Ageing_Image_Classification_with_Convolutional_Neural_Network-CNN-
   ```

2. (Opsional tapi disarankan) Buat virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # Linux/Mac
   ```

3. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

   > Jika file `requirements.txt` belum tersedia, install package inti secara manual:
   > ```bash
   > pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
   > ```

## Cara Penggunaan

### Training Model

Buka dan jalankan notebook `code.ipynb` untuk melakukan preprocessing data, membangun arsitektur CNN, serta melatih model menggunakan dataset pada folder `dataset-final/`.

### Evaluasi Model

Notebook ini juga mencakup evaluasi performa model pada data training dan testing, termasuk:
- Akurasi & loss (training vs testing)
- Confusion matrix (dengan visualisasi heatmap)
- Classification report (precision, recall, f1-score per kelas)

### Inference / Prediksi Gambar Baru

Contoh script untuk memprediksi gambar baru menggunakan model `.h5`:

```python
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Load model
model = load_model("saved_model/model_cnn_facial_skin_ageing.h5")

# Definisikan mapping kelas (sesuaikan dengan urutan kelas model kamu)
idx_to_class = {
    0: "nama_kelas_1",
    1: "nama_kelas_2",
    2: "nama_kelas_3",
}

IMG_SIZE = (224, 224)  # sesuaikan dengan input size model

def select_image():
    root = Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Pilih gambar untuk prediksi",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return img_path

def predict_new_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)

    print(f"Prediksi   : {idx_to_class[predicted_idx]}")
    print(f"Confidence : {predictions[predicted_idx] * 100:.2f}%")

    plt.imshow(img)
    plt.title(f"{idx_to_class[predicted_idx]} ({predictions[predicted_idx]*100:.2f}%)")
    plt.axis("off")
    plt.show()

img_path = select_image()
if img_path:
    predict_new_image(img_path)
```

### Konversi Model

Konversi ke format TensorFlow.js dilakukan melalui notebook `code_convert_model_to_tfjs.ipynb`. Karena package `tensorflowjs` memiliki keterbatasan kompatibilitas dengan versi Python/NumPy terbaru, disarankan menjalankan proses konversi ini di **Google Colab** atau **Kaggle Notebook** dengan environment terpisah (misalnya melalui `condacolab` dengan Python 3.10).

Contoh perintah konversi:
```bash
tensorflowjs_converter --input_format=keras model_cnn_facial_skin_ageing.h5 tfjs_model
```

## Format Model yang Tersedia

| Format | Lokasi Folder | Kegunaan |
|---|---|---|
| `.h5` / `.keras` | `saved_model/` | Load ulang & inference di Python |
| SavedModel (`.pb` + `variables/`) | `saved_model/` | Deployment TensorFlow Serving |
| `.tflite` | `tflite/` | Aplikasi mobile & embedded (Android/iOS) |
| TensorFlow.js (`model.json` + `.bin`) | `tfjs/` | Aplikasi web/browser |

## Kelas Klasifikasi

> **Catatan:** daftar kelas berikut perlu disesuaikan dengan kelas aktual pada dataset (`dataset-final/`) atau file `class_indices.json` / `label.txt` project ini.

| Index | Nama Kelas |
|---|---|
| 0 | *(isi sesuai kelas dataset)* |
| 1 | *(isi sesuai kelas dataset)* |
| 2 | *(isi sesuai kelas dataset)* |

## Requirements

Dependency utama proyek ini:

```
tensorflow
numpy
matplotlib
scikit-learn
seaborn
pillow
```

Untuk kebutuhan konversi ke TensorFlow.js (disarankan pada environment terpisah):
```
tensorflowjs
```

## Catatan Kompatibilitas

- Proses konversi ke TensorFlow.js (`tensorflowjs_converter`) memiliki masalah kompatibilitas dengan **Python 3.12+** dan **NumPy ≥ 1.24** (dependency `tensorflow_hub` dan `read_weights.py` memakai API yang sudah deprecated). Disarankan menjalankan konversi di environment dengan **Python 3.10**, atau menggunakan `condacolab` di Google Colab untuk membuat environment terisolasi.
- Pastikan ukuran input gambar (`IMG_SIZE`) saat inference **sama persis** dengan `input_shape` model — cek dengan `model.input_shape` atau `model.summary()` sebelum menjalankan prediksi.

## Lisensi

Belum ada lisensi resmi yang ditetapkan untuk repository ini. Silakan hubungi pemilik repository untuk informasi penggunaan lebih lanjut.