# 🛡️ ToxiShield: Detecting Toxic Comments Using Logistic Regression and BERT

## 🚀 Project Overview
ToxiShield is an AI-powered system that classifies toxic comments on online platforms. It detects six types of toxic behavior — **toxic**, **severe toxic**, **obscene**, **threat**, **insult**, and **identity hate** — using classical Machine Learning and Deep Learning models.

This project was built as part of the Meta Kaggle Hackathon under the **Main Track**, focusing on **AI for Social Good**.

---

## 📊 Dataset
- **Source:** Jigsaw Toxic Comment Classification Challenge on Kaggle
- **Training Samples:** 159,571
- **Test Samples:** 153,164
- **Labels:** Multi-label classification (six target labels)

---

## 🧰 Methods Used
- ✅ **EDA:** Word Count Distribution, Label Distribution, Word Clouds
- ✅ **Preprocessing:** TF-IDF Vectorization (classical ML), Tokenization (BERT)
- ✅ **Models:**
  - Logistic Regression (Baseline Model)
  - Random Forest Classifier (Ensemble Model)
  - BERT (Bonus Deep Learning Model with Huggingface Transformers)
- ✅ **Evaluation Metrics:** Accuracy, F1-Score, Classification Report

---

## 🏆 Results Summary
| Model                | Toxic Class Accuracy  |          F1-Score        |   Notes                        |
|-----------------------|----------------------|--------------------------|--------------------------------|
| Logistic Regression   | ~95%                 |          91.3%           | Fast, efficient baseline       |
| Random Forest         | ~93%                 |          92.2%           | Robust ensemble method         |
| BERT (Bonus Model)    | High (training sample) |        94.7%           | Powerful language understanding|

---

## 📝 Tech Stack
- Python, scikit-learn, matplotlib, seaborn, wordcloud
- NLP techniques: TF-IDF, Logistic Regression, Random Forest
- BERT (unitary/toxic-bert) via HuggingFace transformers
- Kaggle for execution

---

## 📁 Repository Structure
```
📁 toxishield/
│ ├── 📄 toxic_comment_detection_notebook.ipynb
│ ├── 📄 submission.csv (optional)
│ ├── 📁 report/
│ │   └── toxic_comment_report.md
│ ├── 📁 ppt/
│ │   └── toxic_comment_detection_ppt.pptx
│ └── 📁 images/
│    ├── toxishield_architecture_diagram.png
│    └── wordcloud.png
│    └── screenshots/
│        └── eda.png
│        └── model_results.png
│ 
📄 README.md
📄 LICENSE (MIT)
```
---

## 🚀 How to Reproduce
1. Open [Kaggle Notebook Link](https://www.kaggle.com/your-notebook-link)
2. Run all cells.

## 🔗 Useful Links
- [Dataset](https://www.kaggle.com/competitions/meta-kaggle-hackathon/data)
- [Kaggle Notebook](https://www.kaggle.com/your-notebook-link)
- [GitHub Repository](https://github.com/yourusername/toxishield)

---

## 📣 Credits
Made with ❤️ by Sai Meghana for Meta Hackathon 2025.
