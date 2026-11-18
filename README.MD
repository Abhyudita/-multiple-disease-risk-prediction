# Multiple Disease Risk Prediction System

A machine learning-based web application for predicting disease classification outcomes using health-related measurements.

This project was developed as part of a Bachelor of Technology (B.Tech) capstone project in Computer Science and Engineering (Data Science and Analytics).

## Project Overview

The Multiple Disease Risk Prediction System integrates three machine learning models into a single interactive Streamlit application:

- **Heart Disease Prediction** — Support Vector Machine (SVM)
- **Diabetes Prediction** — XGBoost
- **Chronic Kidney Disease (CKD) Prediction** — Random Forest

The project demonstrates data preprocessing, exploratory analysis, model comparison, cross-validation, model evaluation, and deployment through a web interface.

## Machine Learning Results

| Disease | Selected Model | Test Accuracy |
|---|---|---|
| Heart Disease | SVM | 77.05% |
| Diabetes | XGBoost | 97.26% |
| Chronic Kidney Disease | Random Forest | 100.00% |

These results were obtained on held-out test sets from the respective datasets. They do not establish clinical reliability or generalization to real-world patient populations.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Google Colab
- Jupyter Notebooks

## Project Structure

```text
Multiple_disease_risk_prediction/
├── assets/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
│   └── app.py
├── .gitignore
└── README.md
```

## Model Development

Each disease prediction module was developed using a separate dataset and training workflow.

Models were evaluated using classification metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

Preprocessing and model selection were performed using training data, with separate held-out test sets used for final evaluation.

## Streamlit Application

The application allows users to select one of three disease prediction modules, enter relevant measurements, and view a machine learning classification result.

Predicted probabilities represent model outputs rather than clinically validated estimates of disease risk.

## Limitations

- Models were developed using publicly available datasets.
- Performance may differ on independent populations.
- The CKD dataset contains only 400 records.
- The models have not undergone external clinical validation.
- The application is intended for educational and research demonstration purposes only.

## Contributors

Developed