# 💬 Twitter Sentiment Analysis App

This is a **Streamlit-based web application** that analyzes the sentiment of tweets.  
It uses a trained machine learning model to classify tweets as **Positive**, **Negative**, or **Neutral**.

---

## 🚀 Features
- **Single Tweet Analysis**: Enter a tweet and instantly get its sentiment.
- **Bulk CSV Analysis**: Upload a CSV with a `tweet` or `text` column to analyze multiple tweets.
- **Visualization**: Get a bar chart showing the distribution of sentiments.

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn (for ML model)
- Pandas
- Matplotlib

---

## 📂 Project Structure
twitter-sentiment-analysis/
│── app.py # Main Streamlit app
│── model.pkl # Trained ML model
│── vectorizer.pkl # Text vectorizer
│── assets/banner.jpg # Banner image
│── requirements.txt # Dependencies
│── README.md # Project documentation
---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
2. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py
