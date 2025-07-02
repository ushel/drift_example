
from dotenv import load_dotenv
from pipeline.data_loader import load_data
from pipeline.model import build_model
from pipeline.evaluate import evaluate_model
from pipeline.feature_selection import chi2_feature_selection, anova_feature_selection,preprocess_and_select_numeric_features
# from pipeline.model2 import build_model2
from pipeline.mlflow_tracker import log_experiment, compare_and_promote_model
from sklearn.model_selection import train_test_split
from pipeline.hyperparameter_tuning import tune_hyperparameters
import pandas as pd
import argparse
import os
import numpy as np
import mlflow
import joblib

# CATEGORICAL_COLS = [
#     "Client_Gender", "Client_Marital_Status", "Client_Housing_Type",
#     "Client_Income_Type", "Client_Education", "Loan_Contract_Type"
# ]

# NUMERIC_COLS = [
#     'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days',
#     'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3'
# ]
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
# mlflow.set_tracking_uri("https://dagshub.com/utkarsh.shelke03/loan_deafult_prediction.mlflow")
# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     X_numeric = X[NUMERIC_COLS]
#     num_features = anova_feature_selection(X_numeric, y, k=5)
#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")

#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

#     model = build_model(num_features, cat_features)
#     model.fit(X_train, y_train)
#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model, report, auc)

# if __name__ == "__main__":
#     main()

# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     X_numeric = X[NUMERIC_COLS]
#     num_features = anova_feature_selection(X_numeric, y, k=5)
#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")
#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
#     model = build_model(num_features, cat_features)
#     model.fit(X_train, y_train)
#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model, report, auc)

# if __name__ == "__main__":
#     main()
#  model 1 with hyperparameters


def main(run_name=None):
    df = load_data()
    X = df.drop(columns=["Default"])
    y = df["Default"]

    NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.drop("Default").tolist()
    CATEGORICAL_COLS = df.select_dtypes(include=["object", "category"]).columns.tolist()

    def clean_numeric_column(series):
        cleaned = pd.to_numeric(
            series.astype(str).str.replace(r"[^\d.-]", "", regex=True), errors='coerce'
        )
        if cleaned.dropna().empty:
            return pd.Series([0] * len(series))
        return cleaned

    # Feature selection and preprocessing
    cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
    num_features = preprocess_and_select_numeric_features(df, NUMERIC_COLS)

    selected_features = cat_features + num_features
    X = X[selected_features].copy()

    # Process categorical features
    for col in cat_features:
        X[col] = X[col].astype("category")
        if "missing" not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories(["missing"])
        X[col] = X[col].fillna("missing")

    # Process numeric features
    for col in num_features:
        if col in X.columns:
            X[col] = clean_numeric_column(X[col])
            if X[col].dropna().empty:
                X[col] = 0
            else:
                X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # Ensure categorical dtypes are used by XGBoost
    for col in cat_features:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = tune_hyperparameters(num_features, cat_features, X_train, y_train)

    report, auc = evaluate_model(model, X_test, y_test)
    log_experiment(model, report, auc)
    compare_and_promote_model(report)
    os.makedirs("Data\outputs", exist_ok=True)
    joblib.dump(model, "Data\outputs\model.pkl")

if __name__ == "__main__":
    import traceback
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-name", type=str, default=None)
        args = parser.parse_args()
        main(run_name=args.run_name)
    except Exception as e:
        print("\n🔥 Exception occurred in main.py:")
        traceback.print_exc()
    

# XGBOOST model
# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     num_features = anova_feature_selection(X[NUMERIC_COLS], y, k=5)

#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")
#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

#     model = build_model2(num_features, cat_features)
#     model.fit(X_train, y_train)

#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model.best_estimator_, report, auc)

# if __name__ == "__main__":
#     main()