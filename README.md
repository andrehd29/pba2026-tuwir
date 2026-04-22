# TUGAS BESAR SD25-32202 — Pemrosesan Bahasa Alami
# Multi-Label Sentiment Analysis Spotify Apps

Kelompok 13 (Tuwir):
- Sahid Maulana | 122450109
- Andre Hadiman Rotua Parhusip | 122450108
- Uliano Wilyam | 122450098

Data from Kaggle: (Spotify Reviews Apps Indonesia from Google Play Store)
https://www.kaggle.com/datasets/pandaa12/spotify-reviews-indonesia-google-play-store

Dataset Statistics:
- Total Reviews: 100,000
- Time Period: May 2024 - Dec 2025
- Rating Distribution:
5 Stars: ~67.4% (Positive)
1 Star: ~16.1% (Negative)
- Observation: Negative reviews often mention "Ads" (Iklan), "Premium subscription issues", or "Lyrics not syncing".
- Catatan : Sumber yang digunakan memiliki 100,000 baris data. Akan tetapi, pada proses pengembangan model hanya digunakan 20,000 baris data atau 20% dari total keseluruhan data. Adapaun hal tersebut dilakukan, karena adanya unsur KETERBATASAN DEVICE yang digunakan.

Column Descriptors
The dataset contains the following columns (Headers are in Indonesian):
1. Nama User: Name of the user who posted the review.
2. Ulasan: The text content of the review (Bahasa Indonesia).
3. Rating: The numerical score given by the user (1 to 5 stars).
4. Tanggal: Timestamp of when the review was posted.
5. Likes: Number of thumbs-up votes the review received.
6. Versi App: The version of Spotify installed on the user's device.

Link Deployment Hugging Face : https://huggingface.co/spaces/tubespba-kelompoktuwir/sentimen-spotify-ml
