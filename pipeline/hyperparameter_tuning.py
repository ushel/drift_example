from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from pipeline.preprocessing import get_preprocessor
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os, joblib

def tune_hyperparameters(numeric_cols, categorical_cols, X_train, y_train):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)

    clf = XGBClassifier(
        tree_method='hist',
        device='cuda', 
        use_label_encoder=False,
        eval_metric='logloss',
        enable_categorical=True  # Enables handling of categorical data
    )

    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'reg_lambda': [0.01, 0.05, 0.1, 1, 10],
    }

    sampler = SMOTE(random_state=42)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("sampler", sampler),
        ("classifier", clf)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            'classifier__max_depth': param_grid['max_depth'],
            'classifier__learning_rate': param_grid['learning_rate'],
            'classifier__n_estimators': param_grid['n_estimators'],
            'classifier__subsample': param_grid['subsample'],
            'classifier__reg_lambda': param_grid['reg_lambda'],
        },
        scoring='f1_macro',
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    os.makedirs("Data/outputs", exist_ok=True)
    joblib.dump(best_model, "Data/outputs/model.pkl")
    model_output_path = 'Data/outputs'
    if os.path.exists(model_output_path):
        print(f"Model successfully saved to {model_output_path}")
    else:
        raise FileNotFoundError(f"Model file was not created at {model_output_path}")

    return best_model


# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# def tune_hyperparameters(numeric_cols, categorical_cols, X_train, y_train):
#     classifier = RandomForestClassifier(random_state=42, class_weight="balanced")
#     model = build_model(numeric_cols, categorical_cols, classifier=classifier, sampler_strategy="SMOTEENN")

#     param_grid = {
#         'classifier__n_estimators': [100, 200],
#         'classifier__max_depth': [None, 10, 20],
#         'classifier__min_samples_split': [2, 5]
#     }

#     grid_search = GridSearchCV(model, param_grid, cv=3, scoring='recall', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     return grid_search.best_estimator_
