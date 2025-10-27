import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article text to check if it’s **Real** or **Fake**.")

# ---- Load and train model (simple demonstration) ----
@st.cache_resource
def train_model():
    # Load dataset from online source (small sample)
    data = pd.read_csv("https://raw.githubusercontent.com/dineshreddy9499/fake-news-dataset/main/news.csv")
    x = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    xv_train = vectorizer.fit_transform(x)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(xv_train, y)
    return model, vectorizer

model, vectorizer = train_model()

# ---- Input Section ----
user_input = st.text_area("📝 Paste your news content here:")

if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == "FAKE":
            st.error("🚨 This news seems to be **FAKE or MISLEADING**.")
        else:
            st.success("✅ This news seems to be **REAL and AUTHENTIC**.")

st.markdown("---")
st.caption("Developed as a Mini Project – Fake News Detection using Text Classification")
