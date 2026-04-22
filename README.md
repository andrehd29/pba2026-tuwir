# 🎵 Tugas Besar Pemrosesan Bahasa Alami — Sentiment Analysis Spotify Reviews

> **Benchmarking Machine Learning (Scikit Learn) vs Deep Learning (BiLSTM PyTorch) untuk Klasifikasi Sentimen Ulasan Spotify Berbahasa Indonesia**

Repository ini berisi implementasi tugas besar mata kuliah **SD25-32202 Pemrosesan Bahasa Alami** Semester Genap 2025/2026, Institut Teknologi Sumatera (ITERA).

---

## 👥 Anggota Kelompok

| Nama | NIM | GitHub |
| ---- | --- | ------ |
| Sahid Maulana | 122450109 | @username1 |
| Andre Hadiman Rotua Parhusip | 122450108 | @andrehd29 |
| Uliano Wilyam | 122450098 | @username3 |


---

## 📊 Dataset

**Spotify App Reviews — Indonesian Google Play Store**

- Sumber: Google Play Store (scraped via `google-play-scraper`)
- Total: ~100.000 ulasan berbahasa Indonesia
- Target klasifikasi: 3 kelas sentimen (`Negatif`, `Netral`, `Positif`)
- Mapping label: rating 1-2 → Negatif | rating 3 → Netral | rating 4-5 → Positif

---

## 🚀 Live Demo (Hugging Face Spaces)

| Model | Framework | Link Demo |
| ----- | --------- | --------- |
| 🌲 **Decision Tree + TF-IDF** | scikit-learn | [🔗 ML Model](https://huggingface.co/spaces/tubespba-kelompoktuwir/sentimen-spotify-ml) |
| 🧠 **BiLSTM 2-layer** | PyTorch | [🔗 DL Model](https://huggingface.co/spaces/tubespba-kelompoktuwir/sentimen-spotify-dl) |

> ✏️ **TODO:** Update link DL Model setelah Space DL berhasil di-deploy.

---

## 📄 Paper

| Status | Link |
| ------ | ---- |
| Published on arXiv | [🔗 arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) |

> ✏️ **TODO:** Update link arXiv setelah paper di-publish (Checkpoint 4).

---

## 🗂️ Struktur Repository

```
TUGAS BESAR/
│
├── data/                                   # Dataset asli
│   └── ulasan_com.spotify.music.csv
│
├── ml/                                     # Pendekatan Machine Learning
│   ├── notebooks/
│   │   ├── EDA_PyCaret_Sentimen_Spotify.ipynb
│   │   ├── hasil_benchmark_ml.csv
│   │   └── *.png                           # Visualisasi EDA & hasil
│   └── deployment/                         # Aplikasi Streamlit untuk deploy ke HF Space
│       ├── app.py
│       ├── model_sklearn.pkl
│       ├── tfidf_vectorizer.pkl
│       ├── preprocessing_info.pkl
│       ├── requirements.txt
│       ├── README.md
│       └── runtime.txt
│
├── dl/                                     # Pendekatan Deep Learning
│   ├── notebooks/
│   │   └── DL_BiLSTM_Sentimen_Spotify.ipynb
│   └── deployment/                         # Aplikasi Streamlit untuk deploy ke HF Space
│       ├── app.py
│       ├── model_bilstm.pt
│       ├── vocab.pkl
│       ├── preprocessing_info.pkl
│       ├── requirements.txt
│       └── README.md
│
├── paper/                                  # Manuskrip arXiv (LaTeX)
│   ├── main.tex
│   └── refs.bib
│
└── README.md                               # File ini
```

---

## 🔬 Metodologi

### Machine Learning

- **Preprocessing:** Cleaning → normalisasi slang → stopword removal → stemming (Sastrawi)
- **Feature Extraction:** TF-IDF (`max_features=3000`)
- **Class Balancing:** SMOTE
- **Algoritma yang dibandingkan:** SVM, Multinomial Naive Bayes, Decision Tree
- **Model terbaik:** Decision Tree
- **Dataset:** ~100.000 ulasan (full dataset)

### Deep Learning (BiLSTM)

- **Preprocessing:** Sama dengan ML (untuk fair comparison)
- **Tokenisasi:** Word-level, vocabulary top-10K
- **Arsitektur:**
    - Embedding Layer (vocab=10K, dim=128)
    - BiLSTM 2-layer (hidden=128, bidirectional)
    - FC Layer (256 → 64 → 3) + Dropout
- **Total parameter:** ~2M (≤ 10M ✅, sesuai ToR)
- **Optimizer:** Adam, lr=1e-3, CrossEntropyLoss
- **Training:** Early Stopping, max 10 epoch
- **Dataset:** ~20.000 ulasan (subset stratified)

> **Catatan limitasi:** Model DL dilatih pada subset 20K karena keterbatasan compute lokal (CPU-only). Akan dijelaskan di paper.

---

## 🛠️ Cara Menjalankan Lokal

### Setup Environment

```bash
# Clone repository
git clone https://github.com/[username]/[nama-repo].git
cd [nama-repo]

# Install dependencies (untuk ML)
cd ml/deployment
pip install -r requirements.txt
streamlit run app.py
```

### Training Ulang

**Model ML:**
```bash
cd ml/notebooks
jupyter notebook EDA_PyCaret_Sentimen_Spotify.ipynb
```

**Model DL:**
```bash
cd dl/notebooks
jupyter notebook DL_BiLSTM_Sentimen_Spotify.ipynb
```

---

## 📈 Hasil Benchmark

> ✏️ **TODO:** Akan diupdate setelah model DL selesai dilatih.

| Model | Framework | Accuracy | Precision | Recall | F1-Score | Params |
| ----- | --------- | -------- | --------- | ------ | -------- | ------ |
| Decision Tree | sklearn | TBD | TBD | TBD | TBD | - |
| SVM | sklearn | TBD | TBD | TBD | TBD | - |
| Multinomial NB | sklearn | TBD | TBD | TBD | TBD | - |
| **BiLSTM** | **PyTorch** | **TBD** | **TBD** | **TBD** | **TBD** | **~2M** |

---

## 📚 Referensi

Lihat detail referensi pada paper arXiv (akan tersedia setelah Checkpoint 4).

---

## 📝 Lisensi

Tugas akademik. Penggunaan untuk keperluan edukasi dan riset.

---

**Mata Kuliah:** SD25-32202 Pemrosesan Bahasa Alami
**Dosen Pengampu:** Martin C.T. Manullang
**Institusi:** Institut Teknologi Sumatera (ITERA)
**Semester:** Genap 2025/2026
