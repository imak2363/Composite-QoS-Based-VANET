"""
Corrected explainability run: uses RandomForest, not XGBoost, as the
Composite/none explainability target, since RandomForest (macro-F1 0.6614)
is actually the strongest individual tree-based learner for this scheme,
ahead of XGBoost (0.6548) and DecisionTree (0.6452). The original
run_explainability.py hardcoded XGBoost without checking this.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

RANDOM_STATE = 42
IN_PATH = "/mnt/user-data/outputs/modeling_table_expanded.csv"
OUT_DIR = "/mnt/user-data/outputs"
FEATURE_COLS = ["Nodes", "Speed_kmh", "Flows", "Packet_Rate_pps",
                 "Offered_Load_kbps", "Flows_per_Node", "Rate_x_Flows",
                 "Nodes_x_Speed", "Speed_per_Flow"]

df = pd.read_csv(IN_PATH)
X = df[FEATURE_COLS].values
le = LabelEncoder()
y = le.fit_transform(df["Best_Protocol_Composite"].values)
class_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# best hyperparameters found for Composite | none | RandomForest
model = RandomForestClassifier(
    random_state=RANDOM_STATE, n_jobs=-1, n_estimators=200,
    min_samples_split=8, min_samples_leaf=6, max_features="sqrt", max_depth=None,
)
model.fit(X_train, y_train)
print("Refit RandomForest test accuracy:", model.score(X_test, y_test))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
sv_list = shap_values if isinstance(shap_values, list) else [shap_values[:, :, c] for c in range(shap_values.shape[2])]

importance = pd.DataFrame({class_names[c]: np.abs(sv_list[c]).mean(axis=0)
                            for c in range(len(class_names))}, index=FEATURE_COLS)
importance["Overall"] = importance.mean(axis=1)
importance = importance.sort_values("Overall", ascending=False)
importance.to_csv(f"{OUT_DIR}/shap_top_features.csv")
print(importance)

mean_abs_overall = np.abs(np.stack(sv_list, axis=2)).mean(axis=2)
plt.figure(figsize=(7, 5))
shap.summary_plot(mean_abs_overall, X_test, feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary_composite.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
importance.drop(columns="Overall").plot(kind="barh", ax=ax)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Per-class feature importance (SHAP), Composite scheme, RandomForest")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_bar_composite.png", dpi=200)
plt.close()

lime_explainer = LimeTabularExplainer(
    X_train, feature_names=FEATURE_COLS, class_names=class_names,
    mode="classification", random_state=RANDOM_STATE, discretize_continuous=True)

y_pred_test = model.predict(X_test)
lime_rows = []
for cls_idx, cls_name in enumerate(class_names):
    correct = np.where((y_test == cls_idx) & (y_pred_test == cls_idx))[0]
    candidates = correct if len(correct) > 0 else np.where(y_test == cls_idx)[0]
    idx = candidates[0]
    exp = lime_explainer.explain_instance(X_test[idx], model.predict_proba, num_features=6, top_labels=1)
    pred_label = class_names[model.predict(X_test[idx:idx + 1])[0]]
    top_label_idx = exp.available_labels()[0]
    for feat, weight in exp.as_list(label=top_label_idx):
        lime_rows.append({"true_label": cls_name, "predicted_label": pred_label,
                           "instance_index_in_test": int(idx), "feature_rule": feat, "weight": weight})
    exp.save_to_file(f"{OUT_DIR}/lime_example_{cls_name}.html")

lime_df = pd.DataFrame(lime_rows)
lime_df.to_csv(f"{OUT_DIR}/lime_examples.csv", index=False)
print(lime_df)
print("\nSaved corrected SHAP + LIME outputs (RandomForest) to", OUT_DIR)
