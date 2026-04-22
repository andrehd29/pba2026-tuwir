---
title: Sentimen Spotify ML
emoji: 🎵
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---

# 🎵 Analisis Sentimen Ulasan Spotify

Aplikasi analisis sentimen ulasan Spotify berbahasa Indonesia menggunakan model
**Decision Tree + TF-IDF** (scikit-learn).

## 📚 Konteks Akademik

- **Mata Kuliah:** SD25-32202 Pemrosesan Bahasa Alami
- **Institusi:** Institut Teknologi Sumatera (ITERA)
- **Program Studi:** Sains Data

## 🔬 Pipeline

1. **Preprocessing teks:** lowercasing, hapus URL/mention/hashtag/angka, normalisasi slang Indonesia, stopword removal, stemming (Sastrawi).
2. **Feature extraction:** TF-IDF (`max_features=3000`).
3. **Balancing:** SMOTE.
4. **Model:** Decision Tree Classifier (sklearn 1.4.2).

## 📊 Dataset

~100.000 ulasan aplikasi Spotify dari Google Play Store, diberi label
3 kelas: `Negatif`, `Netral`, `Positif` (mapping dari rating 1-5).

## 🗂️ Struktur File

```
app.py                    # Streamlit app
model_sklearn.pkl         # Decision Tree (terlatih)
tfidf_vectorizer.pkl      # TF-IDF vectorizer
preprocessing_info.pkl    # slang_dict, stopwords, metrics
requirements.txt          # Dependencies (versi pinned)
runtime.txt               # Python 3.11
```
