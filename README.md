# 🎬 IMDB Movie Review Sentiment Analysis (DistilBERT Transformer)

This project is a **movie review sentiment analysis web app** built with **Streamlit** and powered by a **DistilBERT Transformer model** trained on the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).  
It predicts whether a given movie review is **Positive 😊** or **Negative 😞**.

## 🚀 Features
- **Fine-tuned DistilBERT Model:** Leverages the power of Transformers to achieve 92% Test Accuracy.
- **Advanced Tokenization:** Uses Hugging Face AutoTokenizer for robust text processing.
- **Interactive Streamlit UI:** Clean interface with sample positive and negative reviews..
- **Confidence score** Displays the probability confidence for every prediction.
- Optimized with **`@st.cache_resource`** to load the heavy model only once per session..

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/sushrutha777/Movie-Review-Analysis.git
   cd Movie-Sentiment-Analysis

2. Create a new environment and install dependencies:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
