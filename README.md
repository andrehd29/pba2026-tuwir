# 🎵 Tugas Besar Pemrosesan Bahasa Alami — Sentiment Analysis of Indonesian Spotify Reviews Using Machine Learning and BiLSTM

> **Benchmarking Machine Learning (Scikit Learn) vs Deep Learning (BiLSTM PyTorch) untuk Klasifikasi Sentimen Ulasan Spotify Berbahasa Indonesia**

Repository ini berisi implementasi tugas besar mata kuliah **SD25-32202 Pemrosesan Bahasa Alami** Semester Genap 2025/2026, Institut Teknologi Sumatera (ITERA).

---

## 👥 Anggota Kelompok

| Nama | NIM | GitHub |
| ---- | --- | ------ |
| Uliano Wilyam Purba | 122450098 | @uliano122450098 |
| Andre Hadiman Rotua Parhusip | 122450108 | @andrehd29 |
| Sahid Maulana | 122450109 | @beginneraingmah1614 |


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


---

## 📄 Paper

| Status | Link |
| ------ | ---- |
| Submitted to arXiv (Awaiting Publication) | [🔗 arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) |

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
| Decision Tree | sklearn | 72.86% | 72.73% | 72.86% | 72.69% | - |
| SVM | sklearn | 69.91% | 69.56% | 69.91% | 69.58% | - |
| Multinomial NB | sklearn | 49.48% | 61.81% | 49.48% | 46.19% | - |
| **BiLSTM** | **PyTorch** | **83.14%** | **78.41%** | **83.14%** | **80.69%** | **~2M** |

---

## 📚 Referensi

Lihat detail referensi pada paper arXiv.

[1] Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis.
    Foundations and Trends in Information Retrieval, 2(1–2), 1–135.
    https://doi.org/10.1561/1500000011

[2] Pratap, A., & Nidhi. (2020). Comparative sentiment analysis of app reviews.
    arXiv:2006.09739.

[3] Rustam, F., et al. (2021). A comparative study of sentiment analysis using
    NLP and different machine learning techniques on US airline Twitter data.
    arXiv:2110.00859.

[4] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation, 9(8), 1735–1780.
    https://doi.org/10.1162/neco.1997.9.8.1735

[5] Xu, G., Meng, Y., Qiu, X., Yu, Z., & Wu, X. (2019). Sentiment analysis of
    comment texts based on BiLSTM. IEEE Access, 7, 51522–51532.

[6] Lin, C.-H., & Nuha, U. (2023). Sentiment analysis of Indonesian datasets based
    on a hybrid deep-learning strategy. Journal of Big Data, 10(1), 88.
    https://doi.org/10.1186/s40537-023-00782-9

[7] Wilie, B., et al. (2020). IndoNLU: Benchmark and resources for evaluating
    Indonesian natural language understanding. In Proceedings of AACL-IJCNLP 2020,
    pp. 843–857. Association for Computational Linguistics.

[8] Kuncahyo, S., et al. (2020). Improving Bi-LSTM performance for Indonesian
    sentiment analysis using paragraph vector. arXiv:2009.05720.

[9] Farizki, M., et al. (2025). Klasifikasi sentimen menggunakan Bidirectional LSTM
    dan IndoBERT dengan dataset terbatas. ZONAsi: Jurnal Sistem Informasi.

[10] Abiola, O., et al. (2024). Sentiment classification on the 2024 Indonesian
     presidential candidate dataset using deep learning approaches. Indonesian
     Journal of Statistics and Its Applications.

[11] Adriani, M., Asian, J., Nazief, B., Tahaghoghi, S. M. M., & Williams, H. E.
     (2007). Stemming Indonesian. ACM Transactions on Asian Language Information
     Processing, 6(4), 1–33.

[12] Ossai, C. I., & Wickramasinghe, N. (2024). Sentiment analysis on Google Play
     Store app users' reviews based on deep learning approach. Multimedia Tools and
     Applications. https://doi.org/10.1007/s11042-024-19185-w

[13] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
     Synthetic minority over-sampling technique. Journal of Artificial Intelligence
     Research, 16, 321–357.

---

## 📝 Lisensi

Tugas akademik. Penggunaan untuk keperluan edukasi dan riset.

---

**Mata Kuliah:** SD25-32202 Pemrosesan Bahasa Alami
**Dosen Pengampu:** Martin C.T. Manullang
**Institusi:** Institut Teknologi Sumatera (ITERA)
**Semester:** Genap 2025/2026
