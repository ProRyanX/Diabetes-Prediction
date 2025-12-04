# Diabetes-Prediction# 🩺 Diabetes Prediction Using Machine Learning

This repository contains a complete machine learning pipeline to predict diabetes using the **PIMA Indians Diabetes Dataset**.  
Multiple ML models are trained, evaluated, and compared using standard classification metrics.

---

## 📌 Project Overview

Diabetes prediction plays a crucial role in modern healthcare.  
Using medical diagnostic features such as glucose levels, BMI, age, blood pressure, etc., this project builds multiple machine learning models to classify whether a person is diabetic.

The project includes:

- Full ML pipeline  
- EDA (Exploratory Data Analysis)  
- Training 4 ML models  
- Model evaluation  
- Visualizations  
- Feature importance analysis  

---

## 📊 Dataset Information

**Dataset:** PIMA Indians Diabetes Database  
**Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

**Rows:** 768  
**Columns:** 8 features + 1 target  
**Target Column:** `Outcome`  
- `0` → Non-diabetic  
- `1` → Diabetic  

### **Feature Descriptions**

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Patient's age |
| Outcome | Diabetes status (0/1) |

---

## 🎯 Objectives

- Build ML models to predict diabetes.
- Compare Logistic Regression, Decision Tree, Random Forest, and Neural Network.
- Analyze feature importance.
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix

---

## 🧠 Machine Learning Pipeline

1. Load data  
2. Clean & preprocess  
3. Handle zero values  
4. Scale features  
5. Perform EDA  
6. Split into train/test sets  
7. Train multiple ML models  
8. Evaluate and compare performance  
9. Visualize important metrics  
10. Interpret results  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📉 Visualization Section

The following visualizations have been plotted for better understanding of model performance and dataset characteristics:

### ✔ Confusion Matrix  
To analyze the correct and incorrect predictions made by each model.

### ✔ ROC Curve  
To compare the true positive rate against the false positive rate for all models.  
ROC curves help in comparing how well each model is able to separate classes.

### ✔ Feature Importance (Random Forest)  
Shows which features contribute most to the prediction outcome.

### ✔ Neural Network Loss Curve  
Displays how well the MLP model learned during training by showing the loss value per epoch.

---

## 🤖 Models Implemented

| Model | Type |
|-------|------|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based model |
| Random Forest | Ensemble method |
| Neural Network (MLPClassifier) | Deep learning model |

---

## 🧪 Final Evaluation Results

### **📌 Logistic Regression**
- Accuracy: **0.7532**  
- Precision: 0.6545  
- Recall: 0.6545  
- F1 Score: 0.6545  
- ROC-AUC Score: 0.7313  

### **📌 Decision Tree**
- Accuracy: **0.7338**  
- Precision: 0.6167  
- Recall: 0.6727  
- F1 Score: 0.6435  
- ROC-AUC Score: 0.7202  

### **📌 Random Forest**
- Accuracy: **0.7338**  
- Precision: 0.6250  
- Recall: 0.6364  
- F1 Score: 0.6306  
- ROC-AUC Score: 0.7121  

### **📌 MLP Classifier (Neural Network)**
- **Accuracy: 0.7662** (Highest)  
- **Precision: 0.6667**  
- **Recall: 0.6909**  
- **F1 Score: 0.6786** (Highest)  
- **ROC-AUC Score: 0.7495** (Highest)  

---

## 🥇 Best Model Conclusion

### ⭐ **Best Overall Model: MLP Classifier (Neural Network)**

Based on the evaluation results:

- **Highest Accuracy** → 0.7662  
- **Highest F1 Score** → 0.6786  
- **Highest ROC-AUC Score** → 0.7495  
- **Strong Precision & Recall balance**  

The MLP model outperforms Logistic Regression, Decision Tree, and Random Forest in almost every important metric, making it the **most effective model for this dataset**.

---

## 📝 Conclusion

This project demonstrates how machine learning techniques can be applied to a real medical dataset to build a diabetes prediction system.  
With improved training and convergence, the **MLP Classifier achieved the best performance**, proving that neural networks can provide strong predictive power even on small structured datasets.

---

## 👨‍💻 Author

**Rayan Rahat**  
B.Tech – AI & Data Science  
Graphic Era Deemed University  