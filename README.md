# 🎬 IMDB Movie Review Sentiment Analysis (GRU Model)

This project is a **movie review sentiment analysis web app** built with **Streamlit** and powered by a **GRU (Gated Recurrent Unit) neural network** trained on the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).  
It predicts whether a given movie review is **Positive 😊** or **Negative 😞**.

## 🚀 Features
- **Pre-trained GRU model** for sentiment classification.
- **Text preprocessing** that tokenizes and pads reviews.
- **Interactive Streamlit UI** with sample positive and negative reviews.
- **Confidence score** output for each prediction.
- Optimized with **`@st.cache_resource`** to load the model only once per session.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/sushrutha777/Movie-Sentiment-Analysis.git
   cd Movie-Sentiment-Analysis

2. Create a new environment and install dependencies:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
