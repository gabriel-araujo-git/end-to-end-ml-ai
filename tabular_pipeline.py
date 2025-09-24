# tabular_pipeline.py
"""
Pipeline tabular profissional:
- dataset de exemplo: use um CSV com coluna 'target' (binária ou regressão)
- requer: scikit-learn, xgboost, optuna, shap, pandas, numpy
pip install -U scikit-learn xgboost optuna shap pandas numpy
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import optuna
import shap

RANDOM_STATE = 42

def load_data(path: str):
    df = pd.read_csv(path)
    # Assumes target column named 'target'
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def objective(trial, X, y):
    # Hyperparameters to tune
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    preprocessor = build_preprocessor(X)
    model = xgb.XGBClassifier(**params)

    pipe = Pipeline([
        ('preproc', preprocessor),
        ('clf', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def run_optuna(X, y, n_trials=50):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=n_trials, n_jobs=1)
    print("Best trial:", study.best_trial.params, "AUC:", study.best_value)
    return study.best_trial.params

def train_final(X_train, y_train, X_val, y_val, best_params):
    preprocessor = build_preprocessor(X_train)
    model = xgb.XGBClassifier(**best_params)
    pipe = Pipeline([('preproc', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    print(f"Val ROC-AUC: {auc:.4f}")
    return pipe

def explain_model(pipe, X_sample):
    # SHAP explainability (uses underlying xgboost booster)
    # Get transformed features
    preproc = pipe.named_steps['preproc']
    X_trans = preproc.transform(X_sample)
    bst = pipe.named_steps['clf'].get_booster()
    shap_explainer = shap.TreeExplainer(bst)
    shap_values = shap_explainer.shap_values(xgb.DMatrix(X_trans))
    print("SHAP summary (top features):")
    shap.summary_plot(shap_values, X_trans, show=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV com coluna target')
    parser.add_argument('--trials', type=int, default=40)
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    best_params = run_optuna(X_train, y_train, n_trials=args.trials)
    final_pipe = train_final(X_train, y_train, X_val, y_val, best_params)
    joblib.dump(final_pipe, 'model_pipeline.joblib')
    print("Modelo salvo em model_pipeline.joblib")
    # explain_model(final_pipe, X_val.sample(200, random_state=RANDOM_STATE))
