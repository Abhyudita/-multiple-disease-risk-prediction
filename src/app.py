
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Multiple Disease Risk Prediction System",
    page_icon="❤️",
    layout="centered"
)

# Model file locations
BASE_DIR = Path(__file__).resolve().parent.parent

HEART_MODEL_PATH = (
    BASE_DIR / "models" / "heart_disease_pipeline.pkl"
)

DIABETES_PREPROCESSOR_PATH = (
    BASE_DIR / "models" / "diabetes_preprocessor.pkl"
)

DIABETES_XGBOOST_PATH = (
    BASE_DIR / "models" / "diabetes_xgboost.json"
)

CKD_MODEL_PATH = (
    BASE_DIR / "models" / "ckd_pipeline.pkl"
)


@st.cache_resource
def load_heart_model():
    return joblib.load(HEART_MODEL_PATH)


@st.cache_resource
def load_diabetes_components():
    preprocessor = joblib.load(DIABETES_PREPROCESSOR_PATH)

    model = xgb.XGBClassifier()
    model.load_model(DIABETES_XGBOOST_PATH)

    return preprocessor, model

@st.cache_resource
def load_ckd_model():
    return joblib.load(CKD_MODEL_PATH)


st.title("Multiple Disease Risk Prediction System")

st.info(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)


disease = st.selectbox(
    "Select a disease",
    [
        "Heart Disease",
        "Diabetes",
        "Chronic Kidney Disease"
    ]
)

# ----------------------------------------
# HEART DISEASE PREDICTION
# ----------------------------------------

if disease == "Heart Disease":

    st.header("❤️ Heart Disease Risk Prediction")

    st.write(
        "Enter the following health measurements "
        "to generate a prediction using our trained SVM model."
    )

    with st.form("heart_prediction_form"):

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex (dataset encoding)",
            options=[0, 1],
            format_func=lambda x: (
                "Female (0)" if x == 0 else "Male (1)"
            )
        )

        cp = st.selectbox(
            "Chest pain type (dataset code)",
            options=[0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=50,
            max_value=700,
            value=200
        )

        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            )
        )

        restecg = st.selectbox(
            "Resting ECG result (dataset code)",
            options=[0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum heart rate achieved (bpm)",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise-induced angina",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            )
        )

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope of peak exercise ST segment (dataset code)",
            options=[0, 1, 2]
        )

        ca = st.selectbox(
            "Number of major vessels (dataset code)",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia test result (dataset code)",
            options=[0, 1, 2, 3]
        )

        submitted = st.form_submit_button("Predict")

    if submitted:

        feature_values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        input_df = pd.DataFrame([feature_values])

        try:
            model = load_heart_model()

            input_df = input_df[
                model.feature_names_in_
            ]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(
                input_df
            )[0][1]

            if prediction == 1:
                st.warning(
                    "Model prediction: Positive class (1)"
                )
            else:
                st.success(
                    "Model prediction: Negative class (0)"
                )

            st.write(
                "Estimated model probability for class 1: "
                f"{probability:.1%}"
            )

            st.caption(
                "This is a statistical model output, "
                "not an individual's validated clinical risk. "
                "Do not use it to make medical decisions."
            )

        except Exception as error:
            st.error(
                f"Unable to generate prediction: {error}"
            )


# ----------------------------------------
# DIABETES PREDICTION
# ----------------------------------------

elif disease == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    st.write(
        "Enter the following information to generate "
        "a prediction using our trained XGBoost model."
    )

    with st.form("diabetes_prediction_form"):

        gender = st.selectbox(
            "Gender (dataset category)",
            ["Female", "Male", "Other"]
        )

        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=40.0,
            step=1.0
        )

        hypertension = st.selectbox(
            "Hypertension",
            options=[0, 1],
            format_func=lambda x: (
                "No" if x == 0 else "Yes"
            )
        )

        heart_disease = st.selectbox(
            "History of heart disease",
            options=[0, 1],
            format_func=lambda x: (
                "No" if x == 0 else "Yes"
            )
        )

        smoking_history = st.selectbox(
            "Smoking history",
            [
                "No Info",
                "never",
                "former",
                "current",
                "not current",
                "ever"
            ]
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        hba1c = st.number_input(
            "HbA1c level (%)",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1
        )

        blood_glucose = st.number_input(
            "Blood glucose level (mg/dL)",
            min_value=50,
            max_value=500,
            value=140
        )

        submitted = st.form_submit_button(
            "Predict Diabetes"
        )

    if submitted:

        feature_values = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": blood_glucose
        }

        input_df = pd.DataFrame([feature_values])

        try:
            preprocessor, model = load_diabetes_components()

            # Match the original training feature order
            expected_features = [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_history",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level"
            ]

            input_df = input_df[expected_features]

            # Apply the fitted preprocessing steps
            processed_input = preprocessor.transform(input_df)

            # Generate probability using the XGBoost JSON model
            probability = model.predict_proba(
                processed_input
            )[0][1]

            # Use the default 0.50 threshold
            prediction = int(probability >= 0.50)

            if prediction == 1:
                st.warning(
                    "Model prediction: Positive class (1)"
                )
            else:
                st.success(
                    "Model prediction: Negative class (0)"
                )

            st.write(
                "Estimated model probability for class 1: "
                f"{probability:.1%}"
            )

            st.caption(
                "This output is based on a machine learning "
                "model and is not a validated clinical risk "
                "estimate or medical diagnosis. "
                "Do not use it to make medical decisions."
            )

        except Exception as error:
            st.error(
                f"Unable to generate prediction: {error}"
            )

# ----------------------------------------
# CHRONIC KIDNEY DISEASE PREDICTION
# ----------------------------------------

elif disease == "Chronic Kidney Disease":

    st.header("🫘 Chronic Kidney Disease Prediction")

    st.write(
        "Enter the following health measurements "
        "to generate a prediction using our trained "
        "Random Forest model."
    )

    st.caption(
        "The fields below follow the UCI CKD dataset. "
        "Use actual measurements where available."
    )

    with st.form("ckd_prediction_form"):

        st.subheader("General Measurements")

        age = st.number_input(
            "Age (years)", 0, 120, 45
        )

        bp = st.number_input(
            "Blood pressure (mm Hg)", 0, 250, 80
        )

        sg = st.selectbox(
            "Urine specific gravity",
            [1.005, 1.010, 1.015, 1.020, 1.025]
        )

        al = st.selectbox(
            "Urine albumin (dataset level)",
            [0, 1, 2, 3, 4, 5]
        )

        su = st.selectbox(
            "Urine sugar (dataset level)",
            [0, 1, 2, 3, 4, 5]
        )

        st.subheader("Urine Examination")

        rbc = st.selectbox(
            "Red blood cells",
            ["normal", "abnormal"]
        )

        pc = st.selectbox(
            "Pus cells",
            ["normal", "abnormal"]
        )

        pcc = st.selectbox(
            "Pus cell clumps",
            ["notpresent", "present"]
        )

        ba = st.selectbox(
            "Bacteria",
            ["notpresent", "present"]
        )

        st.subheader("Blood Test Measurements")

        bgr = st.number_input(
            "Random blood glucose (mg/dL)",
            min_value=0.0,
            value=120.0
        )

        bu = st.number_input(
            "Blood urea (mg/dL)",
            min_value=0.0,
            value=35.0
        )

        sc = st.number_input(
            "Serum creatinine (mg/dL)",
            min_value=0.0,
            value=1.0
        )

        sod = st.number_input(
            "Sodium (mEq/L)",
            min_value=0.0,
            value=140.0
        )

        pot = st.number_input(
            "Potassium (mEq/L)",
            min_value=0.0,
            value=4.5
        )

        hemo = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=0.0,
            value=13.5
        )

        pcv = st.number_input(
            "Packed cell volume (%)",
            min_value=0.0,
            value=42.0
        )

        wbcc = st.number_input(
            "White blood cell count (cells/cumm)",
            min_value=0.0,
            value=8000.0
        )

        rbcc = st.number_input(
            "Red blood cell count (millions/cumm)",
            min_value=0.0,
            value=4.8
        )

        st.subheader("Medical History and Symptoms")

        htn = st.selectbox(
            "Hypertension",
            ["no", "yes"]
        )

        dm = st.selectbox(
            "Diabetes mellitus",
            ["no", "yes"]
        )

        cad = st.selectbox(
            "Coronary artery disease",
            ["no", "yes"]
        )

        appet = st.selectbox(
            "Appetite",
            ["good", "poor"]
        )

        pe = st.selectbox(
            "Pedal edema",
            ["no", "yes"]
        )

        ane = st.selectbox(
            "Anemia",
            ["no", "yes"]
        )

        submitted = st.form_submit_button(
            "Predict CKD"
        )

    # Generate CKD prediction when the form is submitted
    if submitted:

        feature_values = {
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "su": su,
            "rbc": rbc,
            "pc": pc,
            "pcc": pcc,
            "ba": ba,
            "bgr": bgr,
            "bu": bu,
            "sc": sc,
            "sod": sod,
            "pot": pot,
            "hemo": hemo,
            "pcv": pcv,
            "wbcc": wbcc,
            "rbcc": rbcc,
            "htn": htn,
            "dm": dm,
            "cad": cad,
            "appet": appet,
            "pe": pe,
            "ane": ane
        }

        input_df = pd.DataFrame([feature_values])

        try:
            # Load the trained CKD pipeline
            model = load_ckd_model()

            # Match the original training feature order
            input_df = input_df[
                model.feature_names_in_
            ]

            # Generate prediction and probability
            prediction = model.predict(input_df)[0]

            probability = model.predict_proba(
                input_df
            )[0][1]

            if prediction == 1:
                st.warning(
                    "Model prediction: CKD (positive class)"
                )
            else:
                st.success(
                    "Model prediction: Not CKD (negative class)"
                )

            st.write(
                "Estimated model probability for CKD: "
                f"{probability:.1%}"
            )

            st.caption(
                "This is a machine learning prediction "
                "based on the UCI CKD dataset. "
                "It is not a medical diagnosis or a "
                "clinically validated risk estimate. "
                "Do not use it to make medical decisions."
            )

        except Exception as error:
            st.error(
                f"Unable to generate CKD prediction: {error}"
            )
