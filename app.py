# 🛡️ ToxiShield: Toxic Comment Detector - Streamlit App

import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="ToxiShield: Toxic Comment Detector", page_icon="🛡️", layout="centered")

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
st.sidebar.header("About")
st.sidebar.write("Built with Streamlit, Scikit-learn, and WordCloud for Kaggle Hackathon")

# Text Input
st.header("💬 Enter Comment")
text = st.text_area("Type your comment here:", height=150)

if st.button("🚀 Predict Toxicity"):
    if not text.strip():
        st.warning("Please enter a comment to predict.")
    else:
        input_tfidf = tfidf.transform([text])
        prediction = model.predict(input_tfidf)
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        st.subheader("🎯 Prediction Result:")
        for i, label in enumerate(labels):
            status = "✅ Not Detected" if prediction[0][i]==0 else "⚠️ Detected"
            st.write(f"**{label.capitalize()}:** {status}")

# WordCloud Section
st.header("☁️ Word Cloud of Training Data")
wordcloud_image = "images/wordcloud.png"
st.image(wordcloud_image, caption="Word Cloud", use_column_width=True)

# Optional Bonus Section
st.markdown("---")
st.subheader("📊 Sample EDA Visualizations")
st.image("images/eda.png", caption="EDA Label Distribution", use_column_width=True)
st.image("images/model_results.png", caption="Model Performance Comparison", use_column_width=True)

st.markdown("---")
st.success("Project ToxiShield Completed 🎉")
