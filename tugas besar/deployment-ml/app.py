"""
🎵 Analisis Sentimen Ulasan Spotify — Model ML (PyCaret)
SD25-32202 Pemrosesan Bahasa Alami — ITERA
Deployment: Hugging Face Spaces (Streamlit)
"""

import streamlit as st
import re
import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Spotify",
    page_icon="🎵",
    layout="centered"
)

# ============================================================
# Load Model, TF-IDF Vectorizer & Preprocessing Info
# ============================================================
@st.cache_resource
def load_all_models():
    model = load_model('model_sentimen_spotify_ml')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('preprocessing_info.pkl', 'rb') as f:
        preprocess_info = pickle.load(f)
    return model, tfidf, preprocess_info

model, tfidf, preprocess_info = load_all_models()

slang_dict = preprocess_info['slang_dict']
stopwords_id = set(preprocess_info['stopwords'])

# ============================================================
# Preprocessing Function
# ============================================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    USE_STEMMER = True
except ImportError:
    USE_STEMMER = False

def clean_text(text):
    """Pipeline preprocessing teks ulasan."""
    if not isinstance(text, str) or text.strip() == '':
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in stopwords_id and len(w) > 1]

    if USE_STEMMER:
        words = [stemmer.stem(w) for w in words]

    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# Prediction Function
# ============================================================
def predict_sentiment(text):
    """Prediksi sentimen dari teks ulasan."""
    cleaned = clean_text(text)
    if cleaned == '':
        return None, None, cleaned

    text_tfidf = tfidf.transform([cleaned])
    test_df = pd.DataFrame(text_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    prediction = predict_model(model, data=test_df)
    label = prediction['prediction_label'].iloc[0]
    score = prediction['prediction_score'].iloc[0]

    return label, score, cleaned

# ============================================================
# UI
# ============================================================
st.title("🎵 Analisis Sentimen Ulasan Spotify")
st.markdown("""
### Model Machine Learning dengan PyCaret AutoML
**SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera**

Masukkan ulasan pengguna Spotify dalam Bahasa Indonesia untuk menganalisis sentimennya.
""")

st.divider()

# Input
input_text = st.text_area(
    "📝 Masukkan Ulasan",
    placeholder="Contoh: aplikasi ini sangat bagus dan membantu...",
    height=120
)

# Contoh ulasan
st.markdown("**Contoh ulasan:**")
examples = [
    "suka banget sama spotify, lagunya lengkap dan enak didengar",
    "banyak iklan sangat mengganggu, tidak bisa mendengarkan lagu",
    "lumayan sih tapi masih perlu perbaikan",
    "error terus gabisa dibuka, tolong perbaiki",
    "biasa aja, masih banyak bug",
]

cols = st.columns(3)
for i, ex in enumerate(examples):
    with cols[i % 3]:
        if st.button(ex[:30] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state['input_text'] = ex
            st.rerun()

# Cek jika ada contoh yang diklik
if 'input_text' in st.session_state:
    input_text = st.session_state.pop('input_text')

# Tombol analisis
if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
    if not input_text or input_text.strip() == '':
        st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis sentimen..."):
            label, score, cleaned = predict_sentiment(input_text)

        if label is None:
            st.warning("⚠️ Teks terlalu pendek atau tidak mengandung kata bermakna.")
        else:
            emoji_map = {'Positif': '😊', 'Negatif': '😠', 'Netral': '😐'}
            color_map = {'Positif': '#27ae60', 'Negatif': '#e74c3c', 'Netral': '#f39c12'}

            emoji = emoji_map.get(label, '❓')
            color = color_map.get(label, '#333')

            # Hasil
            st.divider()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; 
                            background-color: {color}15; border-radius: 15px;
                            border: 2px solid {color};">
                    <div style="font-size: 60px;">{emoji}</div>
                    <div style="font-size: 28px; font-weight: bold; color: {color}; margin-top: 10px;">
                        {label}
                    </div>
                    <div style="font-size: 18px; color: #666; margin-top: 5px;">
                        Confidence: {score:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Detail Analisis:**")
                st.markdown(f"**Teks Asli:** {input_text}")
                st.markdown(f"**Setelah Preprocessing:** {cleaned}")
                st.markdown(f"**Prediksi:** {label}")
                st.markdown(f"**Confidence:** {score:.4f}")

                # Progress bar confidence
                st.progress(score, text=f"Confidence: {score:.1%}")

# ============================================================
# Info Model
# ============================================================
st.divider()

with st.expander("📊 Info Model & Performa"):
    model_info = preprocess_info.get('metrics', [])
    best_model_name = preprocess_info.get('model_name', 'N/A')

    st.markdown(f"**Model Terbaik:** {best_model_name}")

    if model_info:
        metrics_df = pd.DataFrame(model_info)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### ℹ️ Tentang
    - **Dataset:** 100.000 ulasan Spotify dari Google Play Store
    - **Preprocessing:** Cleaning, normalisasi slang, stopword removal, stemming (Sastrawi)
    - **Feature Extraction:** TF-IDF (max_features=3000)
    - **Balancing:** SMOTE
    - **Algoritma:** SVM, Multinomial Naive Bayes, Decision Tree
    - **Model Terbaik:** Decision Tree
    - **Framework:** PyCaret AutoML
    """)

st.caption("SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera (ITERA)")
