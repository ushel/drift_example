# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import streamlit as st
# import pandas as pd
# import joblib
# from drift_detector import detect_drift
# import mlflow
# from pipeline.mlflow_tracker import log_inference_metrics
# from pipeline.feature_selection import chi2_feature_selection, anova_feature_selection,preprocess_and_select_numeric_features
# import numpy as np
# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load from .env file manually

# # Explicitly set environment variables for MLflow
# # os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
# # os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# # MLFLOW_TRACKING_USERNAME="utkarshshelke03"
# # MLFLOW_TRACKING_PASSWORD ="55bbb7475062084353130c086369f4602ba5785e"

# st.title("Loan Default Prediction")

# # Load trained model
# model = joblib.load("model.pkl")

# # Define expected column types
# # NUMERIC_COLS = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3']
# # CATEGORICAL_COLS = ['Client_Education', 'Client_Housing_Type', 'Client_Income_Type', 'Client_Marital_Status', 'Loan_Contract_Type', 'Client_Gender']
# # CATEGORICAL_COLS = X.select_dtypes(include="object").columns.tolist()
# # NUMERIC_COLS = X.select_dtypes(include=["number"]).columns.tolist()

# # Load reference data for drift detection
# REFERENCE_DATA_PATH = "Data/Dataset.csv"
# reference_df = pd.read_csv(REFERENCE_DATA_PATH)


# def preprocess_input(input_df):
#     X = input_df.drop(columns=["Default"])
#     y = input_df["Default"]
#     input_df = input_df.copy()
#     # input_df1 = input_df.drop(columns=["Default"])
#     CATEGORICAL_COLS = input_df.select_dtypes(include="object").columns.tolist()
#     NUMERIC_COLS = input_df.select_dtypes(include=["number"]).columns.tolist()

#     # cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     # # num_features = anova_feature_selection(X[NUMERIC_COLS], y, k=5)
#     # num_features = preprocess_and_select_numeric_features(input_df, NUMERIC_COLS)
#     input_df = input_df.copy()
#     # Ensure numeric columns are numeric and imputed
#     # for col in NUMERIC_COLS:
#     #     if col in input_df.columns:
#     #         input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
#     #         median_value = input_df[col].median()
#     #         input_df[col] = input_df[col].fillna(median_value)
#     #     else:
#     #         input_df[col] = 0  # Default value if column missing
#     # for col in NUMERIC_COLS:
#     #     if col in X.columns:
#     #     # Step 1: Force any invalid strings (like 'x') to NaN
#     #         X[col] = pd.to_numeric(X[col], errors='coerce')
#     for col in NUMERIC_COLS:
#         if col in input_df.columns:
#             # Remove non-numeric characters (e.g., commas, symbols)
#             input_df[col] = input_df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
#             input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
#             median_value = input_df[col].median()
#             input_df[col] = input_df[col].fillna(median_value)
#         else:
#             input_df[col] = 0  # Default value if column missing
            
#         # Step 2: Fill NaN with median if possible, otherwise fallback to 0
#         if X[col].dropna().size > 0:
#             X[col].fillna(X[col].median(), inplace=True)
#         else:
#             print(f"Warning: Column '{col}' has no valid numeric values, filling with 0")
#             X[col].fillna(0, inplace=True)        

#     # Ensure categorical columns are string and fill missing values
#     for col in CATEGORICAL_COLS:
#         if col in input_df.columns:
#             input_df[col] = input_df[col].astype(str).fillna("missing")
#         else:
#             input_df[col] = "missing"

#     # Keep only expected columns
#     input_df = input_df[NUMERIC_COLS + CATEGORICAL_COLS]

#     return input_df

# def predict(input_df):
#     input_df = preprocess_input(input_df)
#     probabilities = model.predict_proba(input_df)[:, 1]
#     return probabilities

# def log_to_mlflow(y_true, y_pred, drift_report):
#     with mlflow.start_run(run_name="Streamlit_Inference"):
#         report = classification_report(y_true, y_pred, output_dict=True)
#         mlflow.log_metric("precision", report["1"]["precision"])
#         mlflow.log_metric("recall", report["1"]["recall"])
#         mlflow.log_metric("f1-score", report["1"]["f1-score"])
#         mlflow.log_dict(drift_report, "drift_report.json")

# # THRESHOLD = 0.3  # Lowering threshold increases recall

# # def predict(input_df):
# #     input_df = preprocess_input(input_df)
# #     probabilities = model.predict_proba(input_df)[:, 1]
# #     predictions = (probabilities >= THRESHOLD).astype(int)
# #     return probabilities, predictions

# # Upload CSV
# uploaded_file = st.file_uploader("Upload CSV File for Prediction", type="csv")

# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
#     # try:
#     #     # Preprocess both datasets before drift detection
#     #     ref_processed = preprocess_input(reference_df)
#     #     input_processed = preprocess_input(input_df)

#     #     drift_report = detect_drift(ref_processed, input_processed, log_to_mlflow=False)
#     #     st.subheader("Drift Detection Report")
#     #     st.json(drift_report)

#     #     # Run prediction
#     #     probs = predict(input_df)
#     #     input_df['Default_Probability'] = probs
#     #     st.success("Predictions generated successfully!")
#     #     st.write(input_df)
#     #     if "Default" in input_df.columns:
#     #         y_true = input_df["Default"]
#     #         log_inference_metrics(y_true, preds, drift_report)
#     # except Exception as e:
#     #     st.error(f"Error in prediction: {str(e)}")
#     try:
#         probs = predict(input_df)
#         input_df['Default_Probability'] = probs
#         predictions = (probs >= 0.3).astype(int)

#         y_true = input_df['Default'] if 'Default' in input_df.columns else None

#         # Log inference metrics to MLflow
#         drift_report = detect_drift(reference_df,input_df)
#         log_inference_metrics(y_true=y_true, y_pred=predictions, drift_report=drift_report)

#         if drift_report and drift_report.get("drift_detected"):
#             st.warning("Potential Data Drift Detected")
#             st.json(drift_report)

#         st.success("Predictions generated and logged successfully!")
#         st.write(input_df)

#     except Exception as e:
#         st.error(f"Error in prediction: {str(e)}")


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import pandas as pd
# import joblib
# import numpy as np
# import streamlit as st
# from sklearn.metrics import precision_score, recall_score, f1_score
# from pipeline.mlflow_tracker import log_inference_metrics
# from drift_detector import detect_drift, population_stability_index

# st.title("Loan Default Prediction App")

# # Load trained model
# model = joblib.load("Data\outputs\model.pkl")

# # Define expected columns
# NUMERIC_COLS = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days',
#                 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3']
# CATEGORICAL_COLS = ['Client_Education', 'Client_Housing_Type', 'Client_Income_Type',
#                     'Client_Marital_Status', 'Loan_Contract_Type', 'Client_Gender']

# # Load reference data
# REFERENCE_DATA_PATH = "Data/Dataset.csv"
# reference_df = pd.read_csv(REFERENCE_DATA_PATH)

# def preprocess_input(df):
#     df = df.copy()

#     for col in NUMERIC_COLS:
#         if col in df.columns:
#             df[col] = df[col].replace({',': '', ' ': ''}, regex=True)
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#             df[col] = df[col].fillna(df[col].median())
#         else:
#             df[col] = 0

#     for col in CATEGORICAL_COLS:
#         if col in df.columns:
#             df[col] = df[col].astype(str).fillna("missing")
#         else:
#             df[col] = "missing"

#     return df[NUMERIC_COLS + CATEGORICAL_COLS]

# def predict(df):
#     X = preprocess_input(df)
#     proba = model.predict_proba(X)[:, 1]
#     return proba


# def run_psi_drift_detection(reference, current, threshold=0.2):
#     psi_scores = {
#         col: population_stability_index(reference[col], current[col])
#         for col in NUMERIC_COLS if col in current.columns
#     }
#     drift_detected = any(score > threshold for score in psi_scores.values())
#     return psi_scores, drift_detected

# uploaded_file = st.file_uploader("Upload CSV File", type="csv")

# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)

#     try:
#         probs = predict(input_df)
#         input_df["Default_Probability"] = probs
#         predictions = (probs >= 0.3).astype(int)

#         y_true = input_df["Default"] if "Default" in input_df.columns else None

#         drift_report = detect_drift(reference_df, input_df)

#         log_inference_metrics(
#             y_true=y_true,
#             y_pred=predictions,
#             drift_report=drift_report
#         )

#         st.success(" Prediction and logging successful!")
#         st.dataframe(input_df)

#         if drift_report.get("drift_detected", False):
#             st.warning("Data Drift Detected")
#             st.json(drift_report)

#     except Exception as e:
#         st.error(f" Error: {str(e)}")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import classification_report
from pipeline.mlflow_tracker import log_inference_metrics
from drift_detector import detect_drift, population_stability_index

st.title("Loan Default Prediction App (with PSI Drift Detection)")

# Load trained model
model = joblib.load("Data/outputs/model.pkl")

REFERENCE_DATA_PATH = "Data/Dataset.csv"
reference_df = pd.read_csv(REFERENCE_DATA_PATH, low_memory=False)
reference_df = reference_df.dropna(subset=["Default"])

# Dynamically detect numeric and categorical columns from reference data
NUMERIC_COLS = reference_df.select_dtypes(include=[np.number]).columns.drop("Default").tolist()
CATEGORICAL_COLS = reference_df.select_dtypes(include=["object", "category"]).columns.tolist()

def clean_numeric_column(series):
    cleaned = pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.-]", "", regex=True), errors='coerce'
    )
    if cleaned.dropna().empty:
        return pd.Series([0] * len(series))
    return cleaned

def preprocess_input(df):
    df = df.copy()

    missing_cols = [col for col in NUMERIC_COLS + CATEGORICAL_COLS if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in uploaded data: {missing_cols}")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
            if df[col].dropna().empty:
                df[col] = 0
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")
        else:
            df[col] = "missing"

    return df[NUMERIC_COLS + CATEGORICAL_COLS]

def predict(df):
    X = preprocess_input(df)
    proba = model.predict_proba(X)[:, 1]
    return proba

def run_psi_drift_detection(reference, current, threshold=0.2):
    psi_scores = {}
    for col in NUMERIC_COLS:
        if col in current.columns:
            try:
                score = population_stability_index(reference[col], current[col])
                if np.isfinite(score):
                    psi_scores[col] = score
            except Exception as e:
                psi_scores[col] = None
    drift_detected = any(score is not None and score > threshold for score in psi_scores.values())
    return psi_scores, drift_detected

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("Uploaded columns:", input_df.columns.tolist())

    try:
        probs = predict(input_df)
        input_df["Default_Probability"] = probs
        predictions = (probs >= 0.3).astype(int)

        y_true = input_df["Default"] if "Default" in input_df.columns else None
        if y_true is not None:
            y_true = y_true.astype(int)

        drift_report = detect_drift(reference_df, input_df)
        psi_scores, psi_drift = run_psi_drift_detection(reference_df, input_df)

        if y_true is not None:
            log_inference_metrics(
                y_true=y_true,
                y_pred=predictions,
                drift_report=drift_report
            )

        st.success("Prediction and logging successful.")
        st.dataframe(input_df)

        if drift_report.get("drift_detected", False):
            st.warning("Categorical/Numeric Drift Detected (Stat Test)")
            st.json(drift_report)

        if psi_drift:
            st.warning("PSI Drift Detected in Numeric Features")
            st.json(psi_scores)

    except Exception as e:
        st.error(f"Error: {str(e)}")

