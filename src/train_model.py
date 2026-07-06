"""
Entraînement du modèle de prédiction de churn (Telco Customer Churn).

- Pipeline complet (encodage + scaling + modèle) directement réutilisable par l'app.
- Gestion du déséquilibre : class_weight / scale_pos_weight, comparaison avec SMOTE.
- Tuning par GridSearchCV (scoring ROC-AUC).
- Optimisation du seuil de décision (compromis recall / précision — clé en rétention client).
- Sauvegarde du meilleur pipeline, du seuil, des métriques et des visualisations.

Usage : python src/train_model.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, accuracy_score,
    precision_recall_curve, roc_curve, confusion_matrix,
)
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "telco_customer_churn_clean.csv")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load():
    df = pd.read_csv(DATA)
    df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def make_preprocessor(X):
    numeric = [c for c in NUMERIC if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]), numeric, categorical


def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X, y = load()
    pre, numeric, categorical = make_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Clients: {len(X)} | churn: {y.mean():.1%} | train {len(X_train)} / test {len(X_test)}\n")

    # ── Modèles de base (déséquilibre géré par pondération) ──
    candidates = {
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("pre", pre),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                scale_pos_weight=ratio, eval_metric="logloss", random_state=42)),
        ]),
        "smote_xgboost": ImbPipeline([
            ("pre", pre),
            ("smote", SMOTE(random_state=42)),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                eval_metric="logloss", random_state=42)),
        ]),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)
        m = results[name]
        print(f"{name:20} ROC-AUC={m['roc_auc']:.3f} | recall={m['recall']:.3f} | precision={m['precision']:.3f} | f1={m['f1']:.3f}")

    # ── Tuning du meilleur type de modèle (XGBoost) par GridSearch ──
    print("\nTuning XGBoost (GridSearchCV, scoring ROC-AUC)...")
    grid = GridSearchCV(
        Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
            scale_pos_weight=ratio, eval_metric="logloss", random_state=42))]),
        param_grid={
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.03, 0.05],
            "clf__subsample": [0.8, 1.0],
        },
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    candidates["xgboost_tuned"] = grid.best_estimator_
    results["xgboost_tuned"] = evaluate("xgboost_tuned", grid.best_estimator_, X_test, y_test)
    print(f"  meilleurs params : {grid.best_params_}")
    print(f"  xgboost_tuned        ROC-AUC={results['xgboost_tuned']['roc_auc']:.3f} | recall={results['xgboost_tuned']['recall']:.3f}")

    # ── Sélection du meilleur modèle (ROC-AUC) ──
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = candidates[best_name]
    print(f"\n>>> Meilleur modèle : {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.3f})")

    # ── Optimisation du seuil de décision (maximise le F1) ──
    proba = best_model.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    best_threshold = float(thr[best_idx])
    print(f"Seuil optimal (F1) : {best_threshold:.2f} -> recall={rec[best_idx]:.3f}, precision={prec[best_idx]:.3f}")

    metrics_default = evaluate(best_name, best_model, X_test, y_test, 0.5)
    metrics_opt = evaluate(best_name, best_model, X_test, y_test, best_threshold)

    # ── Sauvegardes ──
    joblib.dump(best_model, os.path.join(MODELS_DIR, "churn_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "comparison": results,
            "best_model": best_name,
            "threshold_default": 0.5,
            "threshold_optimal": best_threshold,
            "metrics_at_0.5": metrics_default,
            "metrics_at_optimal": metrics_opt,
        }, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"numeric": numeric, "categorical": categorical,
                   "columns": list(X.columns), "threshold": best_threshold}, f, indent=2)

    # ── Visualisations ──
    plt.figure(figsize=(7, 6))
    for name in ["logistic_regression", "random_forest", "xgboost_tuned"]:
        p = candidates[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Taux de faux positifs"); plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"), dpi=120); plt.close()

    cm = confusion_matrix(y_test, (proba >= best_threshold).astype(int))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Purples")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Reste", "Churn"]); plt.yticks([0, 1], ["Reste", "Churn"])
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title(f"Matrice de confusion (seuil {best_threshold:.2f})"); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120); plt.close()

    print("\nArtefacts : models/churn_model.joblib, metrics.json, feature_schema.json + reports/*.png")


if __name__ == "__main__":
    main()
