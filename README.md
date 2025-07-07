# Analisis Sentimen Bahasa Indonesia dengan IndoBERT dan Gemini AI

![Image](https://github.com/user-attachments/assets/b2a5984a-4d40-4211-95d5-7f58bf887268)

## 📝 Deskripsi Proyek

Proyek ini adalah aplikasi analisis sentimen teks Bahasa Indonesia yang menggabungkan kekuatan model IndoBERT untuk klasifikasi sentimen dan Gemini AI untuk memberikan penjelasan kontekstual. Aplikasi ini dapat mengklasifikasikan teks dalam Bahasa Indonesia ke dalam tiga kategori sentimen: Positif, Netral, atau Negatif, dilengkapi dengan tingkat kepercayaan dan penjelasan AI yang mudah dipahami.

## ✨ Fitur Utama

- **Klasifikasi Sentimen Otomatis**: Mengidentifikasi sentimen teks (Positif, Netral, Negatif)
- **Tingkat Kepercayaan**: Menampilkan seberapa yakin model dengan prediksinya
- **Penjelasan AI**: Generasi penjelasan kontekstual oleh Gemini AI
- **Antarmuka Pengguna Responsif**: Desain modern yang bekerja di berbagai perangkat
- **Fine-Tuning Model**: Kemampuan untuk melatih ulang model dengan data baru
- **Evaluasi Kinerja**: Laporan metrik evaluasi model yang komprehensif

## 🛠 Teknologi yang Digunakan

### Backend
- **Python 3.10+**
- **Flask**: Framework web backend
- **Transformers (Hugging Face)**: Untuk model IndoBERT
- **PyTorch**: Untuk komputasi deep learning
- **Google Generative AI**: Untuk penjelasan kontekstual
- **Scikit-learn**: Untuk evaluasi model

### Frontend
- **HTML5 & CSS3**: Struktur dan styling antarmuka
- **JavaScript**: Interaktivitas dan komunikasi dengan backend
- **Font Awesome**: Ikon-ikon modern

### Model AI
- **IndoBERT**: Model BERT khusus Bahasa Indonesia yang sudah difine-tune
- **Gemini Flash**: Model generatif untuk penjelasan sentimen

## 🚀 Cara Menjalankan Proyek

### Prasyarat
- Python 3.10 atau lebih baru
- pip (Python package manager)
- Akun Google Cloud dengan API Key untuk Gemini AI

### Instalasi

1. **Clone repositori**
   ```bash
   git clone https://github.com/iwancilibur/Analisis-Sentimen-Bahasa-Indonesia-Menggunakan-Transformer-IndoBERT-dan-Gemini-Ai.git
   cd Analisis-Sentimen-Bahasa-Indonesia-Menggunakan-Transformer-IndoBERT-dan-Gemini-Ai
   ```

2. **Buat environment virtual (disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instal dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Buat file .env**
   Buat file `.env` di root direktori dengan konten:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

6. **Buka di browser**
   Buka `http://localhost:5000` di browser favorit Anda

## 🏗 Struktur Proyek

```
sentiment-analysis-id/
├── app.py                # Aplikasi utama Flask
├── config.py             # Konfigurasi aplikasi
├── evaluate.py           # Skrip evaluasi model
├── fine_tuning.py        # Skrip fine-tuning model
├── utils.py              # Fungsi utilitas
├── requirements.txt      # Dependensi Python
├── data/
│   ├── train.csv         # Data latih
│   └── test.csv          # Data uji
├── models/               # Model yang sudah dilatih
├── static/               # Aset statis (CSS, JS, gambar)
└── templates/
    └── index.html        # Template halaman utama
```

## 📊 Dataset

Proyek ini menggunakan dataset custom yang terdiri dari:
- 80 sampel data latih (`train.csv`)
- 80 sampel data uji (`test.csv`)

Setiap sampel berisi:
- `text`: Teks dalam Bahasa Indonesia
- `label`: 
  - 0 = Positif
  - 1 = Netral
  - 2 = Negatif

## 🧠 Model dan Pelatihan

### IndoBERT Base Model
Model dasar yang digunakan adalah `indobenchmark/indobert-base-p1` dari Hugging Face.

### Fine-Tuning
Model difine-tune dengan:
- 3 epoch pelatihan
- Batch size 8
- Learning rate default dari Hugging Face Trainer
- Evaluasi setiap epoch

### Evaluasi Model
Metrik evaluasi yang digunakan:
- Accuracy
- F1-Score (weighted)
- Precision (untuk kelas Negatif)
- Recall (untuk kelas Positif)
- Classification report lengkap
- Confusion matrix visual

## 🌟 Contoh Penggunaan

1. Masukkan teks Bahasa Indonesia di text area
2. Klik "Analisis Sentimen"
3. Lihat hasil klasifikasi sentimen
4. Baca penjelasan AI tentang analisis tersebut
5. Salin hasil atau lakukan analisis baru

## 📈 Hasil Evaluasi

Contoh hasil evaluasi model:

```
Model: Fine-Tuned IndoBERT
Accuracy: 0.9250
F1-Score: 0.9248
Precision (Negatif): 0.9286
Recall (Positif): 0.9231

Classification Report:
              precision    recall  f1-score   support

     Positif     0.9231    0.9231    0.9231        26
      Netral     0.8889    0.9412    0.9143        17
     Negatif     0.9286    0.8966    0.9123        29

    accuracy                         0.9167        72
   macro avg     0.9135    0.9203    0.9166        72
weighted avg     0.9178    0.9167    0.9168        72
```

## 🤝 Kontribusi

Kontribusi terbuka! Berikut cara berkontribusi:
1. Fork proyek ini
2. Buat branch fitur (`git checkout -b fitur/namafitur`)
3. Commit perubahan (`git commit -m 'Tambahkan beberapa fitur'`)
4. Push ke branch (`git push origin fitur/namafitur`)
5. Buat Pull Request


## ✉️ Kontak

Nama Anda - [@iwancilibur](https://github.com/iwancilibur) - iwancilibur@gmail.com


---

Dibuat dengan ❤️ untuk analisis sentimen Bahasa Indonesia yang lebih baik!
