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
| Model                | Toxic Class Accuracy | Notes                          |
|-----------------------|----------------------|--------------------------------|
| Logistic Regression   | ~95%                 | Fast, efficient baseline       |
| Random Forest         | ~93%                 | Robust ensemble method         |
| BERT (Bonus Model)    | High (training sample) | Powerful language understanding |

---

## 📁 Repository Structure
📁 toxishield/
  ├── 📄 toxic_comment_detection_notebook.ipynb
  ├── 📄 submission.csv
  ├── 📁 report/
   └── 📝 toxic_comment_report.md
  ├── 📁 ppt/
   └── 📄 toxic_comment_detection_ppt.pptx
  ├── 📁 images/
   ├── 📊 toxishield_architecture_diagram.png
   └── 📁 screenshots/
    └── 🖼️ [EDA and result screenshots]
  └── 📁 video/
   └── 🔗 youtube_link.txt

---

## 🎥 Demo Video  
[Watch on YouTube](YOUR_YOUTUBE_LINK_HERE)

---

## 🤝 Team  
Built by [Your Name/Team Name] for Meta Kaggle Hackathon.
