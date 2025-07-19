import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# Page Configuration
st.set_page_config(page_title="ToxiShield: Toxic Comment Detector", page_icon="🛡️", layout="centered")

# Banner
st.image("images/banner.png", use_column_width=True)

# Title and Introduction
st.title("🛡️ ToxiShield: Toxic Comment Detector")
st.markdown("""
This web app predicts toxicity of your input comments using a Logistic Regression model trained on toxic comment data.
""")

# Load Model and Vectorizer
@st.cache_data
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()

# Sidebar Info
st.sidebar.header("ℹ️ Project Info")
st.sidebar.markdown("""
- ✅ Built with Streamlit
- ✅ Trained on Jigsaw Dataset
- ✅ Fast inference with Logistic Regression
- 🖥️ GitHub: [View Code](https://github.com/your-repo)
""")

# Text Input Section
st.header("💬 Enter Comment")

sample_comments = ["You are amazing!", "I hate you", "Go to hell!", "Thank you for your help!", "This article sucks"]
if st.button("🎁 Generate Sample"):
    text = random.choice(sample_comments)
else:
    text = ""
text = st.text_area("Type your comment here:", value=text, height=150)

# Prediction Section
if st.button("🚀 Predict Toxicity"):
    if not text.strip():
        st.warning("Please enter a comment to predict.")
    else:
        input_tfidf = tfidf.transform([text])
        prediction = model.predict(input_tfidf)
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        st.markdown("<h3 style='color:purple;'>🎯 Prediction Summary</h3>", unsafe_allow_html=True)
        for i, label in enumerate(labels):
            if prediction[0][i]==1:
                st.error(f"⚠️ {label.capitalize()}: Detected")
            else:
                st.success(f"✅ {label.capitalize()}: Not Detected")

# WordCloud Section
st.header("☁️ Word Cloud of Training Data")
st.image("images/wordcloud.png", caption="Word Cloud", use_container_width=True)

# EDA and Model Metrics Section
st.markdown("---")
st.subheader("📊 Sample EDA Visualizations")
st.image("images/screenshots/eda.png", caption="EDA Label Distribution", use_container_width=True)
st.image("images/screenshots/model_results.png", caption="Model Performance Comparison", use_container_width=True)

st.markdown("---")
st.success("✅ Project ToxiShield Completed 🎉")
