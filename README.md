   Spam SMS Detection

👨‍💻 **Author:** BOLLU SAI KIRAN  
🆔 **Internship ID:** BY25RY229505  
🏢 **Organization:** CODSOFT  
📅 **Internship Domain:** Machine Learning  

---

## 📘 Project Overview

This project builds a **text classification model** to detect spam messages.  
The model classifies each SMS as **Spam** or **Ham (Legitimate)** using NLP techniques and traditional ML algorithms.

---

## 🧠 Objectives
- Apply Natural Language Processing for text classification.  
- Convert raw SMS text into numerical features using TF-IDF.  
- Evaluate performance of multiple ML algorithms.

---

## ⚙️ Algorithms Used
- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

---

## 📊 Dataset
Dataset: [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/uciml/sms-spam-collection-dataset)  
Contains ~5,500 labeled messages (ham/spam).

---

## 🧩 Data Preprocessing
- Text cleaning: lowercasing, stopword removal, punctuation removal.  
- Tokenization and TF-IDF feature extraction.  
- Train-test split (80/20).

---

## 🚀 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Naive Bayes | **98.5%** | 98% | 97% | 98% |
| Logistic Regression | 97.8% | 97% | 96% | 96% |
| SVM | 98.1% | 98% | 96% | 97% |

✅ **Best Model:** Multinomial Naive Bayes with TF-IDF features.

---

## 📦 Libraries Used
- pandas  
- numpy  
- scikit-learn  
- nltk  
