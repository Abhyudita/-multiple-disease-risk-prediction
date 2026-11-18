# Multiple Disease Risk Prediction System

A machine learning-based web application that integrates predictive models for Heart Disease, Diabetes, and Chronic Kidney Disease (CKD) into a single interactive platform.

Developed as a Bachelor of Technology (B.Tech) capstone project in Computer Science and Engineering (Data Science and Analytics).

## Project Overview

The Multiple Disease Risk Prediction System demonstrates the application of machine learning techniques to healthcare-related classification problems.

The project integrates three independently developed machine learning models into a unified Streamlit web application, allowing users to enter health-related measurements and obtain model-generated classification results.

### Disease Prediction Modules

| Disease | Selected Machine Learning Model |
|---|---|
| Heart Disease | Support Vector Machine (SVM) |
| Diabetes | XGBoost |
| Chronic Kidney Disease | Random Forest |

## Machine Learning Results

The selected models were evaluated using held-out test datasets.

| Disease | Selected Model | Test Accuracy |
|---|---|---:|
| Heart Disease | SVM | 77.05% |
| Diabetes | XGBoost | 97.26% |
| Chronic Kidney Disease | Random Forest | 100.00% |

Additional evaluation metrics included:
- Precision
- Recall
- F1 Score
- ROC-AUC

**Note:** These results reflect performance on the respective experimental test sets. They do not establish clinical reliability or generalization to real-world patient populations.

## Project Workflow

The project follows a machine learning development workflow:

1. **Data Collection:** Obtaining publicly available datasets for the three prediction tasks.
2. **Exploratory Data Analysis:** Examining dataset characteristics, feature distributions, missing values, and class distributions.
3. **Data Preprocessing:** Preparing numerical and categorical features for model training.
4. **Model Development:** Training and comparing multiple machine learning algorithms.
5. **Model Evaluation:** Assessing performance using classification metrics and validation techniques.
6. **Model Integration:** Connecting trained models to a unified Streamlit application.
7. **Deployment:** Developing an interactive web interface for demonstrating model predictions.

## Technologies Used

**Programming Language**
- Python

**Data Analysis and Processing**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- XGBoost
- Joblib

**Application Development**
- Streamlit

**Development Environment**
- Google Colab
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
Multiple_disease_risk_prediction/
│
├── assets/
│
├── data/
│   └── raw/
│
├── models/
│   ├── ckd_pipeline.pkl
│   ├── diabetes_preprocessor.pkl
│   ├── diabetes_xgboost.json
│   └── heart_disease_pipeline.pkl
│
├── src/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

Follow these steps to run the application locally.

**1. Clone the repository**

```bash
git clone https://github.com/Abhyudita/-multiple-disease-risk-prediction.git
```

**2. Navigate to the project directory**

```bash
cd -multiple-disease-risk-prediction
```

**3. Create a virtual environment (recommended)**

```bash
python -m venv .venv
```

**4. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Streamlit application**

```bash
streamlit run src/app.py
```

Open the local URL displayed in the terminal to access the application.

## Limitations

- The models were developed using publicly available datasets.
- Model performance may vary across different populations and clinical settings.
- The CKD dataset contains only 400 records, limiting the scope of generalization.
- The models have not undergone independent external clinical validation.
- Predictions are statistical model outputs, not clinically validated individual risk estimates.
- The application is intended strictly for educational and research demonstration purposes.

**Medical Disclaimer:** This application is not a medical diagnostic tool. Its predictions must not be used for medical decisions, diagnosis, or treatment.

## Contributors

- Abhyudita Pandey
- Nishka Sharma
- Raghav Sharma

Bachelor of Technology (B.Tech)  
Computer Science and Engineering – Data Science and Analytics  
DIT University, Dehradun

---

*This project represents an academic exploration of machine learning applications in healthcare and the integration of predictive models into an interactive web application.*
