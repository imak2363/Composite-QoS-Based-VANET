"""
Full VANET routing-protocol-selection classification pipeline.

For each of 2 label schemes (Composite, PDR-only) x 3 imbalance strategies
(none, class-weighting, SMOTE) x 5 classifiers (LogisticRegression,
DecisionTree, RandomForest, XGBoost, Stacking), this:

  1. Splits the modeling table 80/20, stratified by the label column.
  2. Tunes hyperparameters with RandomizedSearchCV (5-fold StratifiedKFold,
     macro-F1 scoring, 20 iterations), applying the imbalance strategy
     correctly within each training fold (SMOTE via an imblearn Pipeline so
     it never touches the validation fold or the test set; class-weighting
     via sample_weight computed on the training split only).
  3. Refits the best configuration on the full training split and evaluates
     once on the held-out test split: accuracy, macro precision/recall/F1,
     one-vs-rest macro ROC-AUC, and the confusion matrix.

Outputs (all under /mnt/user-data/outputs/):
  - test_set_results.csv        one row per (label_scheme, imbalance, model)
  - best_hyperparameters.csv    one row per (label_scheme, imbalance, model)
  - confusion_matrices.csv      long-format confusion-matrix cell counts
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import time
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
IN_PATH = "/mnt/user-data/outputs/modeling_table_expanded.csv"
OUT_DIR = "/mnt/user-data/outputs"

FEATURE_COLS = ["Nodes", "Speed_kmh", "Flows", "Packet_Rate_pps",
                 "Offered_Load_kbps", "Flows_per_Node", "Rate_x_Flows",
                 "Nodes_x_Speed", "Speed_per_Flow"]

LABEL_SCHEMES = {
    "Composite": "Best_Protocol_Composite",
    "PDR": "Best_Protocol_PDR",
}
IMBALANCE_STRATEGIES = ["none", "weighting", "smote"]
N_ITER = 20
CV_FOLDS = 5


def get_param_distributions(model_name):
    if model_name == "LogisticRegression":
        return {
            "clf__C": np.logspace(-3, 2, 30),
        }
    if model_name == "DecisionTree":
        return {
            "clf__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
            "clf__min_samples_split": [2, 4, 6, 8, 10, 15, 20],
            "clf__min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "clf__criterion": ["gini", "entropy"],
        }
    if model_name == "RandomForest":
        return {
            "clf__n_estimators": [100, 200, 300, 400, 500],
            "clf__max_depth": [4, 6, 8, 10, 12, None],
            "clf__min_samples_split": [2, 4, 6, 8, 10],
            "clf__min_samples_leaf": [1, 2, 4, 6],
            "clf__max_features": ["sqrt", "log2", None],
        }
    if model_name == "XGBoost":
        return {
            "clf__n_estimators": [100, 200, 300, 400, 500],
            "clf__max_depth": [2, 3, 4, 5, 6, 8],
            "clf__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "clf__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
        }
    if model_name == "Stacking":
        return {
            "clf__rf__n_estimators": [100, 200, 300],
            "clf__rf__max_depth": [6, 8, 10, None],
            "clf__xgb__n_estimators": [100, 200, 300],
            "clf__xgb__max_depth": [3, 4, 5, 6],
            "clf__xgb__learning_rate": [0.05, 0.1, 0.15],
            "clf__dt__max_depth": [4, 6, 8, None],
            "clf__final_estimator__C": np.logspace(-2, 2, 15),
        }
    raise ValueError(model_name)


def build_estimator(model_name, n_classes):
    if model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ])
    if model_name == "DecisionTree":
        return Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    if model_name == "RandomForest":
        return Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    if model_name == "XGBoost":
        return Pipeline([("clf", XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0,
        ))])
    if model_name == "Stacking":
        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        base_xgb = XGBClassifier(objective="multi:softprob", num_class=n_classes,
                                  eval_metric="mlogloss", random_state=RANDOM_STATE,
                                  n_jobs=-1, verbosity=0)
        base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
        final_lr = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
        stack = StackingClassifier(
            estimators=[("rf", base_rf), ("xgb", base_xgb), ("dt", base_dt)],
            final_estimator=final_lr,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        return Pipeline([("clf", stack)])
    raise ValueError(model_name)


def wrap_with_smote(pipeline: Pipeline) -> ImbPipeline:
    steps = list(pipeline.steps)
    steps.insert(0, ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
    return ImbPipeline(steps)


def main():
    df = pd.read_csv(IN_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    results_rows = []
    hyperparam_rows = []
    cm_rows = []

    for scheme_name, label_col in LABEL_SCHEMES.items():
        X = df[FEATURE_COLS].values
        y_raw = df[label_col].values

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = le.classes_
        n_classes = len(class_names)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for imbalance in IMBALANCE_STRATEGIES:
            for model_name in ["LogisticRegression", "DecisionTree", "RandomForest",
                                "XGBoost", "Stacking"]:
                t0 = time.time()
                base_pipeline = build_estimator(model_name, n_classes)
                param_dist = get_param_distributions(model_name)
                fit_params = {}

                if imbalance == "none":
                    estimator = base_pipeline
                elif imbalance == "weighting":
                    estimator = base_pipeline
                    sw = compute_sample_weight("balanced", y_train)
                    fit_params["clf__sample_weight"] = sw
                elif imbalance == "smote":
                    estimator = wrap_with_smote(base_pipeline)
                else:
                    raise ValueError(imbalance)

                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_dist,
                    n_iter=N_ITER,
                    scoring="f1_macro",
                    cv=cv,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train, **fit_params)

                best_est = search.best_estimator_
                y_pred = best_est.predict(X_test)
                y_proba = best_est.predict_proba(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0
                )
                y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
                try:
                    auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
                except Exception:
                    auc = np.nan

                cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
                for i, true_cls in enumerate(class_names):
                    for j, pred_cls in enumerate(class_names):
                        cm_rows.append({
                            "label_scheme": scheme_name, "imbalance": imbalance,
                            "model": model_name, "true_label": true_cls,
                            "predicted_label": pred_cls, "count": int(cm[i, j]),
                        })

                elapsed = time.time() - t0
                results_rows.append({
                    "label_scheme": scheme_name, "imbalance": imbalance, "model": model_name,
                    "accuracy": acc, "macro_precision": prec, "macro_recall": rec,
                    "macro_f1": f1, "macro_roc_auc": auc,
                    "cv_best_score_macro_f1": search.best_score_,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "seconds": round(elapsed, 1),
                })
                hyperparam_rows.append({
                    "label_scheme": scheme_name, "imbalance": imbalance, "model": model_name,
                    "best_params": json.dumps(search.best_params_),
                })
                print(f"[{scheme_name} | {imbalance:9s} | {model_name:18s}] "
                      f"acc={acc:.4f} macroF1={f1:.4f} macroAUC={auc:.4f} "
                      f"cvF1={search.best_score_:.4f} ({elapsed:.1f}s)")

    results_df = pd.DataFrame(results_rows)
    hyperparams_df = pd.DataFrame(hyperparam_rows)
    cm_df = pd.DataFrame(cm_rows)

    results_df.to_csv(os.path.join(OUT_DIR, "test_set_results.csv"), index=False)
    hyperparams_df.to_csv(os.path.join(OUT_DIR, "best_hyperparameters.csv"), index=False)
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrices.csv"), index=False)

    print("\nSaved test_set_results.csv, best_hyperparameters.csv, confusion_matrices.csv")


if __name__ == "__main__":
    main()
