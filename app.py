import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Page Configuration
st.set_page_config(page_title="ToxiShield: Toxic Comment Detector", page_icon="🛡️", layout="centered")

# Banner
st.image("images/banner.png", use_container_width=True)
st.caption("Empowering Safer Conversations 🛡️")

# Title and Introduction
st.title("🛡️ ToxiShield: Toxic Comment Detector")
st.markdown("""
This web app predicts toxicity of your input comments using a Logistic Regression model trained on toxic comment data.
""")
st.markdown("**💬 Real-time Toxicity Detector — Fast ⚡ Accurate 🎯 Reliable 🛡️**")

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
- 🖥️ GitHub: [View Code](https://github.com/SaiMeghana14/ToxiShield-Toxic-Comment-Detection)
""")

model_choice = st.sidebar.radio("Choose Model", ["Logistic Regression"], index=0)

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

# Extra Chart for Label Distribution (Static Example)
st.subheader("📈 Toxic Label Distribution (Example)")
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
counts = [1200, 300, 800, 150, 700, 100]  # Example counts
fig, ax = plt.subplots()
sns.barplot(x=labels, y=counts, palette="rocket", ax=ax)
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Expander for Dataset Info
with st.expander("See Dataset Details 📊"):
    st.write("Jigsaw Toxic Comment Dataset with multi-label classification.")

st.markdown("---")
st.success("✅ Project ToxiShield Completed 🎉")
st.markdown("<p style='text-align: center;'>Made with ❤️ by Your Name</p>", unsafe_allow_html=True)
