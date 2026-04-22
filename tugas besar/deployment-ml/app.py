"""
🎵 Analisis Sentimen Ulasan Spotify — Model ML (sklearn)
SD25-32202 Pemrosesan Bahasa Alami — ITERA
Deployment: Hugging Face Spaces (Streamlit)
"""

import re
import pickle
import streamlit as st
import pandas as pd

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
    with open('model_sklearn.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('preprocessing_info.pkl', 'rb') as f:
        preprocess_info = pickle.load(f)
    return model, tfidf, preprocess_info

model, tfidf, preprocess_info = load_all_models()

slang_dict = preprocess_info['slang_dict']
stopwords_id = set(preprocess_info['stopwords'])

# Label mapping: model.classes_ = [0, 1, 2]
LABEL_MAP = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

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
        return None, None, cleaned, None

    X = tfidf.transform([cleaned])
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    score = float(proba[pred_class])

    label = LABEL_MAP.get(pred_class, str(pred_class))
    proba_dict = {LABEL_MAP[c]: float(proba[i]) for i, c in enumerate(model.classes_)}

    return label, score, cleaned, proba_dict

# ============================================================
# UI
# ============================================================
st.title("🎵 Analisis Sentimen Ulasan Spotify")
st.markdown("""
### Model Machine Learning — Decision Tree + TF-IDF
**SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera**

Masukkan ulasan pengguna Spotify dalam Bahasa Indonesia untuk menganalisis sentimennya.
""")

st.divider()

# ============================================================
# Session State Init
# Pakai key 'review_text' untuk session state, BEDA dari widget key
# ============================================================
if 'review_text' not in st.session_state:
    st.session_state['review_text'] = ''

# Contoh ulasan — render TOMBOL DULU sebelum text_area
# supaya ketika diklik, value bisa dipakai sebagai default text_area
st.markdown("**📝 Contoh ulasan (klik untuk mengisi otomatis):**")
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
            st.session_state['review_text'] = ex
            st.rerun()

# Text area — pakai value=, BUKAN key= yang sama dengan state
input_text = st.text_area(
    "📝 Masukkan Ulasan",
    value=st.session_state['review_text'],
    placeholder="Contoh: aplikasi ini sangat bagus dan membantu...",
    height=120,
)

# Tombol analisis
if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
    if not input_text or input_text.strip() == '':
        st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis sentimen..."):
            label, score, cleaned, proba_dict = predict_sentiment(input_text)

        if label is None:
            st.warning("⚠️ Teks terlalu pendek atau tidak mengandung kata bermakna.")
        else:
            emoji_map = {'Positif': '😊', 'Negatif': '😠', 'Netral': '😐'}
            color_map = {'Positif': '#27ae60', 'Negatif': '#e74c3c', 'Netral': '#f39c12'}

            emoji = emoji_map.get(label, '❓')
            color = color_map.get(label, '#333')

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
                st.markdown(f"**Setelah Preprocessing:** _{cleaned}_")
                st.markdown(f"**Prediksi:** {label}")
                st.markdown(f"**Confidence:** {score:.4f}")

                if proba_dict:
                    st.markdown("**Probabilitas Kelas:**")
                    proba_df = pd.DataFrame({
                        'Kelas': list(proba_dict.keys()),
                        'Probabilitas': list(proba_dict.values())
                    })
                    st.bar_chart(proba_df.set_index('Kelas'), height=180)

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
    - **Dataset:** ~100.000 ulasan Spotify dari Google Play Store
    - **Preprocessing:** Cleaning, normalisasi slang, stopword removal, stemming (Sastrawi)
    - **Feature Extraction:** TF-IDF (max_features=3000)
    - **Balancing:** SMOTE
    - **Algoritma:** SVM, Multinomial Naive Bayes, Decision Tree
    - **Model Terbaik:** Decision Tree
    - **Framework:** scikit-learn
    """)

st.caption("SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera (ITERA)")
