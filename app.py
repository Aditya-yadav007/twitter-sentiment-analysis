import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Set Streamlit page configuration
st.set_page_config(page_title="Twitter Sentiment App", layout="centered")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"

# ---------- LANDING PAGE ----------
def landing_page():
    st.markdown("<h1 style='text-align: center;'>💬 Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    st.image("assets/banner.jpg", use_container_width=True)

    st.markdown("""
    <div style='text-align: center;'>
        <p style='font-size: 18px;'>
            Discover the power of AI in understanding public opinion. <br><br>
            This tool analyzes Twitter posts to determine whether the sentiment is <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Start Analysis"):
        st.session_state.page = "analysis"

# ---------- ANALYSIS PAGE ----------
def analysis_page():
    st.title("🧠 Twitter Sentiment Analysis")

    # Selection mode
    analysis_mode = st.selectbox(
        "Choose Analysis Mode",
        ["Single Tweet Analysis", "Bulk Analysis from CSV"]
    )

    # ----------------- SINGLE INPUT -----------------
    if analysis_mode == "Single Tweet Analysis":
        tweet_input = st.text_area("✍️ Enter a Tweet", placeholder="Type or paste a tweet here...")

        if st.button("Analyze Tweet"):
            if tweet_input.strip() == "":
                st.warning("⚠️ Please enter a tweet before analyzing.")
            else:
                tweet_vector = vectorizer.transform([tweet_input])
                prediction = model.predict(tweet_vector)[0]

                if prediction == 1:
                    st.success("😊 Positive Sentiment")
                elif prediction == 0:
                    st.info("😠 Negative Sentiment")
                else:
                    st.error("😐 Neutral Sentiment")

    # ----------------- BULK INPUT -----------------
    elif analysis_mode == "Bulk Analysis from CSV":
        uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # Try to find a suitable column
            text_column = None
            for col in df.columns:
                if 'tweet' in col.lower() or 'text' in col.lower():
                    text_column = col
                    break

            if not text_column:
                st.error("❌ Could not find a tweet/text column. Please make sure your CSV includes one.")
            else:
                # Vectorize and predict
                tweets = df[text_column].astype(str)
                tweet_vectors = vectorizer.transform(tweets)
                predictions = model.predict(tweet_vectors)

                # Add results to DataFrame
                sentiment_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
                df['Sentiment'] = predictions
                df['Sentiment_Label'] = df['Sentiment'].map(sentiment_map)

                st.success(f"✅ Analyzed {len(df)} tweets successfully!")
                st.dataframe(df[[text_column, 'Sentiment_Label']])

                # Plot
                st.subheader("📊 Sentiment Distribution")
                sentiment_counts = df['Sentiment_Label'].value_counts()

                fig, ax = plt.subplots()
                colors = ['#1f77b4', '#ff7f0e', '#d62728']  # Positive, Neutral, Negative
                sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                ax.set_ylabel("Number of Tweets")
                ax.set_xlabel("Sentiment")
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)

    # Navigation back
    if st.button("🔙 Back to Home"):
        st.session_state.page = "landing"
# ---------- MAIN ----------
if st.session_state.page == "landing":
    landing_page()
else:
    analysis_page()
