"""
🎵 Analisis Sentimen Ulasan Spotify — BiLSTM (Deep Learning)
SD25-32202 Pemrosesan Bahasa Alami — ITERA
Deployment: Hugging Face Spaces (Streamlit + PyTorch)
"""

import re
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Spotify — DL (BiLSTM)",
    page_icon="🎵",
    layout="centered"
)

# ============================================================
# Model Architecture (HARUS sama persis dengan training)
# ============================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_concat = torch.cat([h_forward, h_backward], dim=1)
        out = self.dropout1(h_concat)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        return self.fc2(out)

# ============================================================
# Load Model, Vocab, Preprocessing Info
# ============================================================
DEVICE = torch.device('cpu')  # HF Space free tier = CPU

@st.cache_resource
def load_all():
    # Load checkpoint
    ckpt = torch.load('model_bilstm.pt', map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']

    model = BiLSTMClassifier(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load vocab
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)

    # Load preprocessing info
    with open('preprocessing_info.pkl', 'rb') as f:
        prep = pickle.load(f)

    return model, vocab_data, prep, cfg

model, vocab_data, preprocess_info, model_config = load_all()
word2idx = vocab_data['word2idx']
IDX_TO_LABEL = vocab_data['idx_to_label']
MAX_LEN = vocab_data['max_len']

slang_dict = preprocess_info['slang_dict']
stopwords_id = set(preprocess_info['stopwords'])

# ============================================================
# Preprocessing (HARUS sama dengan training)
# ============================================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    USE_STEMMER = True
except ImportError:
    USE_STEMMER = False

def clean_text(text):
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
    return re.sub(r'\s+', ' ', ' '.join(words)).strip()

def text_to_indices(text, word2idx, max_len):
    PAD_IDX = word2idx.get('<PAD>', 0)
    UNK_IDX = word2idx.get('<UNK>', 1)
    tokens = text.split()[:max_len]
    indices = [word2idx.get(t, UNK_IDX) for t in tokens]
    if len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))
    return indices

# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def predict_sentiment(text):
    cleaned = clean_text(text)
    if cleaned == '':
        return None, None, cleaned, None

    indices = text_to_indices(cleaned, word2idx, MAX_LEN)
    x = torch.tensor([indices], dtype=torch.long).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = IDX_TO_LABEL[pred_idx]
    score = float(probs[pred_idx])
    proba_dict = {IDX_TO_LABEL[i]: float(probs[i]) for i in range(len(probs))}

    return label, score, cleaned, proba_dict

# ============================================================
# UI
# ============================================================
st.title("🎵 Analisis Sentimen Ulasan Spotify")
st.markdown("""
### Deep Learning — BiLSTM (PyTorch)
**SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera**

Masukkan ulasan pengguna Spotify dalam Bahasa Indonesia untuk menganalisis sentimennya.
""")

st.divider()

# Init state — pakai key BERBEDA dari widget (lesson learned dari ML)
if 'review_text_dl' not in st.session_state:
    st.session_state['review_text_dl'] = ''

# Contoh ulasan
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
        if st.button(ex[:30] + "...", key=f"ex_dl_{i}", use_container_width=True):
            st.session_state['review_text_dl'] = ex
            st.rerun()

input_text = st.text_area(
    "📝 Masukkan Ulasan",
    value=st.session_state['review_text_dl'],
    placeholder="Contoh: aplikasi ini sangat bagus dan membantu...",
    height=120,
)

if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
    if not input_text or input_text.strip() == '':
        st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis sentimen dengan BiLSTM..."):
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

with st.expander("📊 Info Arsitektur Model"):
    total_params = sum(p.numel() for p in model.parameters())
    st.markdown(f"""
    **Arsitektur:** BiLSTM 2-layer + FC head
    **Framework:** PyTorch
    **Total Parameter:** {total_params:,}
    **Constraint ToR:** ≤ 10M parameter ✅

    ### Struktur Layer
    ```
    Embedding(vocab={model_config['vocab_size']:,}, dim={model_config['embed_dim']})
        ↓
    BiLSTM(hidden={model_config['hidden_dim']}, layers={model_config['num_layers']}, bidirectional)
        ↓
    Dropout({model_config['dropout']})
        ↓
    Linear({model_config['hidden_dim']*2} → 64) + ReLU
        ↓
    Dropout(0.3)
        ↓
    Linear(64 → {model_config['num_classes']}) + Softmax
    ```

    ### Tentang
    - **Dataset:** ~20.000 ulasan Spotify (subset stratified dari ~100K)
    - **Preprocessing:** Cleaning, normalisasi slang, stopword removal, stemming (Sastrawi)
    - **Tokenisasi:** Word-level, vocab top-10K, padding ke {MAX_LEN} token
    - **Training:** Adam optimizer, CrossEntropyLoss, Early Stopping (patience=3)
    """)

st.caption("SD25-32202 Pemrosesan Bahasa Alami — Institut Teknologi Sumatera (ITERA)")
