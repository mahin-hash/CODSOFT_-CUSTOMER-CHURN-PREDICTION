# 📊 Customer Churn Prediction
This project uses machine learning to predict whether a customer will churn (leave a service). It trains and evaluates three models using historical customer data.

## 🚀 Features
- Preprocesses customer demographic and usage data
- Trains Logistic Regression, Random Forest, and Gradient Boosting models
- Evaluates model performance using precision, recall, and F1-score
- Saves the best model (`GradientBoostingClassifier`) for future use

## 📁 Dataset
Make sure your dataset file (e.g.,'churn_dataa.csv`) is in the same folder as the script. The target column used is `churn`

## 📦 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
```bash
python churn_model.py
```
This will:

Train all 3 models

Print evaluation results

Save the best model to churn_model.pkl

## 🧠 Sample Output 
```bash
🔍 Training Gradient Boosting...

📊 Classification Report for Gradient Boosting:

              precision    recall  f1-score   support

           0       0.87      0.96      0.91
           1       0.73      0.44      0.55

accuracy: 0.85
✅ Best model saved to churn_model.pkl
```
## 📁 Output Files
churn_model.pkl — Trained ML model for prediction
