# Tugas Besar ETS PCD: Sistem Pengenalan Wajah dan Deteksi Suku

Proyek ini merupakan bagian dari tugas praktikum Pengolahan Citra Digital (PCD) yang bertujuan untuk membangun sistem pengenalan wajah dan deteksi suku di Indonesia. Sistem ini menggunakan **MobileNetV2** untuk klasifikasi suku (Ambon, Batak, Cina, Jawa, Padang, Sunda) dan **FaceNet** untuk perbandingan kesamaan wajah. Dataset wajah diproses dengan deteksi wajah otomatis menggunakan **MTCNN**, cropping wajah, dan penyimpanan ke struktur folder sesuai label suku. Aplikasi ini dibangun menggunakan **Streamlit** sebagai antarmuka pengguna.

## Fitur Utama

- **Tambah Data Wajah**: Unggah gambar wajah, pilih suku melalui dropdown, deteksi wajah otomatis menggunakan MTCNN, crop wajah, dan simpan ke dataset sesuai label suku.
- **Deteksi Suku dari File**: Unggah gambar untuk mendeteksi suku menggunakan model MobileNetV2.
- **Deteksi Suku dari Kamera**: Gunakan kamera untuk mengambil gambar wajah dan mendeteksi suku secara real-time.
- **Daftar Wajah Dataset**: Lihat daftar gambar wajah yang tersimpan di dataset per suku.
- **Face Similarity**: Bandingkan dua gambar wajah untuk menentukan kesamaan menggunakan FaceNet.

## Anggota Kelompok

- Fauzan Rizky Ramadhan (231511076)
- Muhammad Wildan Gumilang (231511087)
- Restu Akbar (231511088)

## Struktur Folder

```
PCD_TUBES_ETS_Deteksi_Suku/
├── data/
│   ├── augmented_train/      # Gambar hasil augmentasi untuk pelatihan
│   ├── processed/            # Gambar wajah yang sudah diproses (cropped)
│   ├── raw/                  # Gambar wajah asli sebelum diproses
│   └── temp/                 # Folder sementara untuk gambar dari kamera/file
├── model/
│   ├── accuracy_plot.png     # Grafik akurasi pelatihan
│   ├── confusion_matrix.png  # Matriks konfusi hasil evaluasi
│   ├── loss_plot.png         # Grafik loss pelatihan
│   ├── mobilenetv2_best.h5   # Model terbaik setelah pelatihan
│   ├── mobilenetv2_final.h5  # Model akhir setelah pelatihan awal
│   └── mobilenetv2_final_finetuned.h5 # Model akhir setelah fine-tuning
├── src/
│   ├── images/               # Gambar untuk UI (foto anggota kelompok)
│   └── model/
│       ├── face_similarity.py  # Kode untuk perbandingan kesamaan wajah
│       ├── predict_suku.py     # Kode untuk prediksi suku
│       ├── train_model.py      # Kode untuk melatih model MobileNetV2
│       └── preprocessing/
│           ├── preprocess.py   # Kode untuk deteksi dan pemrosesan wajah
│           ├── augment.py      # Kode untuk augmentasi data (opsional)
│           ├── config.py       # Konfigurasi (opsional)
│           └── main.py         # Aplikasi utama Streamlit
├── .gitignore                # File untuk mengabaikan file/folder saat git push
├── README.md                 # Dokumentasi proyek (file ini)
└── requirements.txt          # Daftar dependensi
```

## Cara Setup

1. **Clone Repository**:

   ```bash
   git clone <url-repo-anda>
   cd PCD_TUBES_ETS_Deteksi_Suku
   ```

2. **Buat Virtual Environment** (opsional, tetapi disarankan):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   ```

3. **Install Dependensi**: Pastikan file `requirements.txt` ada di direktori proyek, lalu jalankan:

   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan Dataset**:

   - Pastikan folder `data/processed/` berisi gambar wajah yang sudah diproses (cropped) untuk setiap suku (`ambon`, `batak`, `cina`, `jawa`, `padang`, `sunda`).
   - Jika belum ada, gunakan fitur "Tambah Data" pada aplikasi untuk menambah gambar wajah ke folder `data/raw/` dan proses ke `data/processed/`.

5. **Siapkan Model**:

   - Jika Anda belum memiliki model terlatih, jalankan pelatihan terlebih dahulu (lihat "Melatih Model").
   - Pastikan file model `mobilenetv2_best.h5` ada di folder `model/`.

## Cara Menjalankan Aplikasi

1. **Jalankan Aplikasi Streamlit**:

   ```bash
   streamlit run src/model/main.py
   ```

   Aplikasi akan terbuka di browser Anda (biasanya di `http://localhost:8501`).

2. **Gunakan Fitur Aplikasi**:

   - Pilih menu dari sidebar (Home, Tambah Data, Deteksi Suku dari File, Deteksi Suku Sekarang, Daftar Wajah Dataset, Face Similarity).
   - Ikuti instruksi di masing-masing menu untuk menggunakan fitur.

## Melatih Model (Opsional)

Jika Anda ingin melatih ulang model MobileNetV2:

1. Pastikan dataset ada di folder `data/processed/`.

2. Jalankan script pelatihan:

   ```bash
   python src/model/train_model.py
   ```

3. Model yang dilatih akan disimpan di folder `model/` (seperti `mobilenetv2_best.h5`, `mobilenetv2_final.h5`, dll.).

4. Hasil pelatihan (grafik akurasi, loss, matriks konfusi) juga akan disimpan di folder `model/`.

## Dependensi

Semua dependensi ada di file `requirements.txt`. Beberapa library utama yang digunakan:

- `tensorflow==2.15.0`: Untuk pelatihan dan inferensi model MobileNetV2.
- `streamlit`: Untuk antarmuka pengguna.
- `mtcnn`: Untuk deteksi wajah.
- `facenet-pytorch`: Untuk perbandingan kesamaan wajah.
- `opencv-python`: Untuk pemrosesan gambar.
- `scikit-learn`: Untuk evaluasi model (laporan klasifikasi, matriks konfusi).

## Catatan

- Pastikan Anda memiliki koneksi internet saat pertama kali menginstall dependensi.
- Jika Anda menggunakan CPU untuk pelatihan, proses mungkin memakan waktu lebih lama dibandingkan GPU.
- Dataset awal proyek ini hanya memiliki 168 gambar (28 per suku), yang cukup kecil. Untuk performa lebih baik, tambah data (idealnya 50–100 gambar per suku).