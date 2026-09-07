# Facial Skin Ageing Image Classification with CNN

Proyek **Computer Vision** untuk klasifikasi citra kondisi penuaan kulit wajah (*facial skin ageing*) ke dalam 5 kelas menggunakan **Convolutional Neural Network (CNN)** yang dibangun dari nol (from scratch) dengan TensorFlow/Keras. Model kemudian diekspor ke berbagai format (`.keras`, `.h5`, SavedModel, TensorFlow Lite, dan TensorFlow.js) agar bisa digunakan di Python, aplikasi mobile, maupun aplikasi web.


## Daftar Isi

- [Ringkasan Proyek](#ringkasan-proyek)
- [Dataset](#dataset)
- [Alur Pengerjaan (Pipeline)](#alur-pengerjaan-pipeline)
- [Arsitektur Model](#arsitektur-model)
- [Konfigurasi Training](#konfigurasi-training)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Struktur Repository](#struktur-repository)
- [Format Model yang Tersedia](#format-model-yang-tersedia)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Requirements / Library yang Digunakan](#requirements--library-yang-digunakan)
- [Catatan Kompatibilitas](#catatan-kompatibilitas)
- [Lisensi](#lisensi)

## Ringkasan Proyek

Model dilatih untuk mengklasifikasikan gambar wajah ke dalam 5 kategori kondisi kulit terkait penuaan:

- `acne` (jerawat)
- `clear_face` (wajah bersih/normal)
- `darkspots` (flek hitam)
- `puffy_eyes` (mata bengkak/kantung mata)
- `wrinkles` (kerutan)

## Dataset

- **Sumber:** [Kaggle — `siddheshbhatt/facial-skin-ageing`](https://www.kaggle.com/datasets/siddheshbhatt/facial-skin-ageing), diunduh otomatis via `kagglehub.dataset_download()`.
- **Struktur asli:** `dataset/skin_ageing_symptoms/<nama_kelas>/`

**Jumlah gambar per kelas (data asli sebelum augmentasi):**

| Kelas | Jumlah Gambar |
|---|---|
| acne | 4.367 |
| clear_face | 4.600 |
| darkspots | 4.434 |
| puffy_eyes | 3.811 |
| wrinkles | 4.548 |

**Augmentasi data** dilakukan langsung ke folder dataset asli menggunakan kombinasi transformasi acak (rotasi, flip, warp/shift, blur, brightness, shear) untuk menambah variasi & menyeimbangkan jumlah data, dengan target tambahan sebanyak 4.000 gambar untuk kelas `acne`, `clear_face`, `darkspots`, `wrinkles`, dan 2.500 gambar untuk kelas `puffy_eyes`.

**Jumlah gambar final setelah split train/test (80:20)** — disalin ke folder `dataset-final/`:

| Kelas | Train | Test |
|---|---|---|
| acne | 3.562 | 923 |
| clear_face | 3.765 | 1.001 |
| darkspots | 3.651 | 917 |
| puffy_eyes | 3.241 | 994 |
| wrinkles | 3.713 | 1.009 |

## Alur Pengerjaan (Pipeline)

Seluruh proses ada pada satu notebook, `code.ipynb`, dengan tahapan sebagai berikut:

1. **Import Library** — TensorFlow/Keras, OpenCV, scikit-image, PIL, scikit-learn, dsb.
2. **Loading Dataset** — download dataset dari Kaggle via `kagglehub`.
3. **Data Preprocessing**
   - Eksplorasi data (jumlah gambar per kelas, visualisasi sampel gambar, distribusi kelas)
   - Augmentasi gambar kustom (`imageAugmentation` class): rotasi, flip, warp shift, blur, brightness, shear
   - Penggabungan data hasil augmentasi ke DataFrame, lalu di-*shuffle*
4. **Splitting Data** — split 80:20 (train:test), lalu file disalin ke struktur folder `dataset-final/train/<kelas>` dan `dataset-final/test/<kelas>`.
5. **Modeling**
   - `ImageDataGenerator` dengan rescaling (`1/255`) dan `validation_split=0.2` (sehingga data train dipecah lagi menjadi training & validation)
   - Membangun CNN custom (lihat [Arsitektur Model](#arsitektur-model))
   - Training dengan *class weighting* untuk menangani ketidakseimbangan kelas, serta callback `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
   - Evaluasi akurasi/loss training vs testing, confusion matrix, dan classification report
6. **Simpan Model (SavedModel)** — ekspor ke format TensorFlow SavedModel dan HDF5 (`.h5`), serta menyimpan `class_indices.json`.
7. **Convert ke TensorFlow.js** — konversi dari SavedModel ke `tfjs_model/`.
8. **Convert ke TensorFlow Lite** — konversi ke `.tflite`, beserta `label.txt` berisi urutan kelas.
9. **Inferensi (Demo)** — memuat model `.h5`, memilih gambar via dialog file (Tkinter), lalu menampilkan prediksi kelas beserta confidence per kelas.

## Arsitektur Model

Model `model_CNN` adalah CNN sekuensial (`Sequential`) dengan 4 blok konvolusi:

```
Input (150, 150, 3)
├── Conv2D(32, 3x3, relu, padding=same) → BatchNorm → MaxPool(2x2)
├── Conv2D(64, 3x3, relu, padding=same) → BatchNorm → MaxPool(2x2)
├── Conv2D(128, 3x3, relu, padding=same) → BatchNorm → MaxPool(2x2)
├── Conv2D(128, 3x3, relu, padding=same) → BatchNorm → MaxPool(2x2)
├── Flatten
├── Dense(128, relu) → Dropout(0.5)
├── Dense(64, relu)  → Dropout(0.3)
└── Dense(5, softmax)   # 5 kelas output
```

- **Optimizer:** Adam
- **Loss:** `sparse_categorical_crossentropy`
- **Metric:** accuracy

## Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Ukuran gambar input | 150 × 150 × 3 (RGB) |
| Batch size | 32 |
| Epoch (maksimum) | 30 |
| Class mode | `sparse` |
| Rescaling | `1/255.0` |
| Validation split | 0.2 (diambil dari data train) |
| Class weighting | Ya, dihitung otomatis berdasarkan jumlah gambar per kelas |
| Early Stopping | `monitor="val_loss"`, `patience=10`, `restore_best_weights=True` |
| Model Checkpoint | `monitor="val_accuracy"`, `save_best_only=True` |
| Reduce LR on Plateau | `monitor="val_loss"`, `factor=0.5`, `patience=4`, `min_lr=1e-6` |

## Hasil Evaluasi

Hasil aktual dari run notebook (`code.ipynb`):

| Metrik | Training | Testing |
|---|---|---|
| Akurasi | 98,83% | **93,35%** |
| Loss | 4,29% | 24,83% |

**Classification report (data test, 4.844 gambar):**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| acne | 0,9442 | 0,9707 | 0,9573 | 923 |
| clear_face | 0,9608 | 0,9301 | 0,9452 | 1.001 |
| darkspots | 0,9233 | 0,9455 | 0,9343 | 917 |
| puffy_eyes | 0,9305 | 0,8893 | 0,9095 | 994 |
| wrinkles | 0,9103 | 0,9356 | 0,9228 | 1.009 |
| **Accuracy (overall)** | | | **0,9335** | 4.844 |
| Macro avg | 0,9338 | 0,9342 | 0,9338 | 4.844 |
| Weighted avg | 0,9338 | 0,9335 | 0,9334 | 4.844 |

Kelas dengan performa terbaik adalah `acne` (recall 97,07%), sementara `puffy_eyes` sedikit lebih rendah (recall 88,93%) — kemungkinan karena kemiripan visual antar kondisi kulit di sekitar mata.

## Struktur Repository

```
.
├── code.ipynb                              # Notebook utama: preprocessing, augmentasi, training, evaluasi, export model
├── dataset/                                # Dataset mentah hasil download dari Kaggle
│   └── skin_ageing_symptoms/
│       ├── acne/  ├── clear_face/  ├── darkspots/  ├── puffy_eyes/  └── wrinkles/
├── dataset-final/                          # Dataset hasil augmentasi & split 80:20
│   ├── train/<kelas>/
│   └── test/<kelas>/
├── model_cnn_facial_skin_ageing.keras      # Model terbaik hasil ModelCheckpoint (format Keras native)
├── saved_model/                            # Model hasil export TensorFlow SavedModel + HDF5
│   ├── saved_model.pb
│   ├── variables/
│   ├── facial_skin_ageing_model_cnn.h5
│   └── class_indices.json                  # Urutan/nama kelas
├── tfjs_model/                             # Model hasil konversi ke TensorFlow.js
│   ├── model.json
│   └── group1-shard1of2.bin, group1-shard2of2.bin
├── tflite/                                 # Model hasil konversi ke TensorFlow Lite
│   ├── model.tflite
│   ├── facial_skin_ageing_model_cnn.tflite
│   └── label.txt
└── nenek_insidius.jpg                      # Contoh gambar untuk testing inference
```

## Format Model yang Tersedia

| Format | Lokasi | Kegunaan |
|---|---|---|
| `.keras` | `model_cnn_facial_skin_ageing.keras` (root) | Load ulang & inference di Python (format Keras native, hasil checkpoint terbaik) |
| `.h5` | `saved_model/facial_skin_ageing_model_cnn.h5` | Load ulang & inference di Python (dipakai pada demo inferensi notebook) |
| SavedModel (`.pb` + `variables/`) | `saved_model/` | Deployment TensorFlow Serving |
| TensorFlow Lite (`.tflite`) | `tflite/` | Aplikasi mobile & embedded (Android/iOS) |
| TensorFlow.js (`model.json` + `.bin`) | `tfjs_model/` | Aplikasi web/browser |

Urutan kelas (index → label) untuk seluruh format model, tersimpan di `saved_model/class_indices.json` dan `tflite/label.txt`:

```
0: acne
1: clear_face
2: darkspots
3: puffy_eyes
4: wrinkles
```

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

3. Install dependency utama (repository ini belum menyertakan `requirements.txt`):
   ```bash
   pip install tensorflow keras kagglehub numpy pandas seaborn matplotlib scikit-learn opencv-python scikit-image pillow tqdm
   ```

   Untuk konversi ke TensorFlow.js (dijalankan pada bagian akhir `code.ipynb`):
   ```bash
   pip install tensorflowjs
   ```

## Cara Penggunaan

### Training Model

Buka dan jalankan `code.ipynb` secara berurutan mulai dari import library hingga penyimpanan model. Notebook akan otomatis:
1. Mengunduh dataset dari Kaggle,
2. Melakukan augmentasi & split data ke `dataset-final/`,
3. Melatih CNN, dan
4. Mengekspor model ke seluruh format (`saved_model/`, `tfjs_model/`, `tflite/`).

### Evaluasi Model

Evaluasi (akurasi/loss training vs testing, confusion matrix, classification report) sudah termasuk dalam bagian **4. Modeling** di `code.ipynb`.

### Inference / Prediksi Gambar Baru

Contoh kode inferensi (diambil dari bagian **Inferensi (Demo)** pada `code.ipynb`), menggunakan model `.h5`:

```python
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("./saved_model/facial_skin_ageing_model_cnn.h5")

# Load urutan kelas
with open("./saved_model/class_indices.json", "r") as f:
    class_name = json.load(f)

IMG_SIZE = (150, 150)  # harus sama dengan input size model

def select_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    img_path = filedialog.askopenfilename(
        title="Pilih gambar untuk prediksi",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")],
    )
    root.destroy()
    return img_path

def preprocess_image(img_path, target_size=IMG_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img

def predict_new_image(img_path):
    img_array, original_img = preprocess_image(img_path)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_label = class_name[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    print(f"Prediksi   : {predicted_label}")
    print(f"Confidence : {confidence:.2f}%")

    plt.imshow(original_img)
    plt.title(f"{predicted_label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_label, confidence, predictions

img_path = select_image()
if img_path:
    predict_new_image(img_path)
```

### Konversi Model

Konversi ke TensorFlow.js dan TensorFlow Lite sudah dilakukan langsung di dalam `code.ipynb` (bagian **6** dan **7**), tepat setelah model disimpan sebagai SavedModel:

```python
# Ke TensorFlow.js
import tensorflowjs as tfjs
tfjs.converters.convert_tf_saved_model("./saved_model", "./tfjs_model")

# Ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("./saved_model")
tflite_model = converter.convert()
```

## Requirements / Library yang Digunakan

Berdasarkan cell import pada `code.ipynb`:

```
tensorflow
keras
kagglehub
numpy
pandas
seaborn
matplotlib
scikit-learn
opencv-python (cv2)
scikit-image (skimage)
pillow (PIL)
tqdm
```

Untuk konversi model ke TensorFlow.js:
```
tensorflowjs
```

## Catatan Kompatibilitas

- Judul notebook menyebut "MobileNetv2", tetapi model final yang dilatih dan disimpan adalah **CNN custom (Sequential)**, bukan transfer learning MobileNetV2. Modul `MobileNet` dan `DenseNet121` memang di-*import* di awal notebook namun tidak digunakan pada arsitektur final.
- Proses konversi ke TensorFlow.js (`tensorflowjs_converter`) berpotensi mengalami masalah kompatibilitas pada **Python 3.12+** dan **NumPy versi terbaru**. Notebook menyertakan patch kompatibilitas manual (`np.object = object`, `np.bool = bool`) sebelum melakukan konversi.
- Pastikan ukuran input gambar (`IMG_SIZE = (150, 150)`) saat inferensi **sama persis** dengan `input_shape` model — cek dengan `model.input_shape` atau `model.summary()` sebelum menjalankan prediksi.
- Fungsi pemilihan gambar pada demo inferensi menggunakan `tkinter` (dialog file), sehingga hanya dapat dijalankan pada environment lokal dengan tampilan GUI (tidak berjalan di Google Colab/Kaggle tanpa modifikasi).

## Lisensi

Belum ada lisensi resmi yang ditetapkan untuk repository ini. Silakan hubungi pemilik repository untuk informasi penggunaan lebih lanjut.

---

*README ini diperbarui berdasarkan isi aktual `code.ipynb` dan struktur file repository per September 2026.*
