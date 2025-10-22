 Credit Card Fraud Detection

👨‍💻 **Author:** BOLLU SAI KIRAN  
🆔 **Internship ID:** BY25RY229505  
🏢 **Organization:** CODSOFT  
📅 **Internship Domain:** Machine Learning  

---

## 📘 Project Overview

This project focuses on detecting **fraudulent credit card transactions** using machine learning algorithms.  
The dataset contains various anonymized transaction features, and the task is to classify each transaction as **fraudulent** or **legitimate**.

---

## 🧠 Objectives
- Identify fraudulent transactions in credit card data.
- Compare performance of multiple ML algorithms.
- Handle **imbalanced data** effectively using resampling techniques.

---

## ⚙️ Algorithms Used
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

## 📊 Dataset
Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
Contains features `V1`–`V28`, `Amount`, `Time`, and target variable `Class` (0 = Legit, 1 = Fraud).

---

## 🧩 Data Preprocessing
- Handled class imbalance using **SMOTE**.  
- Normalized numerical features using **StandardScaler**.  
- Split data into training and testing sets (80/20).  

---

## 🚀 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Logistic Regression | 98.8% | 92% | 84% | 88% |
| Decision Tree | 99.0% | 95% | 87% | 91% |
| Random Forest | **99.3%** | **96%** | **93%** | **94%** |

✅ **Best Model:** Random Forest Classifier

---

## 📦 Libraries Used
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- imbalanced-learn  
