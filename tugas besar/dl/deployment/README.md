---
title: Sentimen Spotify DL BiLSTM
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.44.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🎵 Analisis Sentimen Ulasan Spotify — BiLSTM (Deep Learning)

Aplikasi klasifikasi sentimen ulasan Spotify berbahasa Indonesia menggunakan model
**BiLSTM 2-layer** (PyTorch) — pembanding terhadap model ML (Decision Tree + TF-IDF).

## 📚 Konteks Akademik

- **Mata Kuliah:** SD25-32202 Pemrosesan Bahasa Alami
- **Institusi:** Institut Teknologi Sumatera (ITERA)
- **Program Studi:** Sains Data
- **Tugas:** Tugas Besar — Benchmark ML vs DL untuk klasifikasi teks

## 🧠 Arsitektur Model

```
Embedding(vocab=10K, dim=128)              → ~1.28M params
       ↓
BiLSTM(hidden=128, layers=2, bidirectional) → ~0.66M params
       ↓
Dropout(0.5)
       ↓
Linear(256 → 64) + ReLU                    → ~16K params
       ↓
Dropout(0.3)
       ↓
Linear(64 → 3)                             → ~195 params
       ↓
Softmax → [Negatif, Netral, Positif]
```

**Total parameter: ~2M** (jauh di bawah constraint ToR 10M ✅)

## 🔬 Pipeline

1. **Preprocessing teks:** lowercasing, hapus URL/mention/hashtag/angka, normalisasi slang Indonesia, stopword removal, stemming (Sastrawi).
2. **Tokenisasi:** word-level, vocabulary top-10K dari training set.
3. **Padding:** ke panjang max sequence (95th percentile).
4. **Model:** BiLSTM 2-layer dengan FC classification head.
5. **Training:** Adam optimizer, CrossEntropyLoss, Early Stopping.

## 📊 Dataset

~20.000 ulasan aplikasi Spotify dari Google Play Store (subset stratified dari ~100K).
Label: `Negatif`, `Netral`, `Positif` (mapping dari rating 1-5).

**Catatan:** Subset dipilih karena keterbatasan compute lokal (CPU-only). Limitasi ini akan dijelaskan di paper.

## 🗂️ Struktur File

```
app.py                    # Streamlit app
model_bilstm.pt           # PyTorch state_dict + config
vocab.pkl                 # word2idx, idx_to_label, max_len
preprocessing_info.pkl    # slang_dict, stopwords (sama dengan ML)
requirements.txt          # PyTorch CPU + Streamlit
```
