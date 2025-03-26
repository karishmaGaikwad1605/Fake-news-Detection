import nltk
nltk.download("stopwords")
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
st.markdown('<style>.stApp {background-image: url("https://media.istockphoto.com/id/1385227287/photo/wooden-blocks-with-real-and-fake-text-of-concept-a-pen-a-notebook-and-a-cup.webp?a=1&b=1&s=612x612&w=0&k=20&c=VqOhWyZ0O3I-sa0qHiQx6z5NorN0M57upaIbi6utc7o="); background-size: cover;}</style>', unsafe_allow_html=True)

# Load the trained model and vectorizer
model = pickle.load(open("modelfr.pkl", "rb"))
vectorizer = pickle.load(open("vectorized.pkl", "rb"))


# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)


# Streamlit App UI
st.markdown("<h1 style='text-align: center; font-size: 36px;'>📰 <b>Real/Fake News Detection</b></h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter a news article or headline to check if it's <b>real</b> or <b>fake</b>.</h3>", unsafe_allow_html=True)

# Text input
user_input = st.text_area("📝 **Paste the news article here:**", "")

if st.button("🔍 **Check News**"):


    if user_input:
        preprocessed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(transformed_text)

        if prediction[0] == 0:
            st.error("🚨 This news is **FAKE**!")
        else:
            st.success("✅ This news is **REAL**!")
    else:
        st.warning("Please enter a news article to check.")

