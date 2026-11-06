"""Phishing Website Detection: four-model comparison with group-aware tuning.

The test fold is held out from all hyperparameter selection. GridSearchCV uses
StratifiedGroupKFold on the training partition only. The positive class is
Phishing (1). Results and figures are saved to phishing_results_tuned/.
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

RANDOM_STATE = 42
N_SPLITS = 5
OUT = Path("phishing_results_tuned")
OUT.mkdir(exist_ok=True)


def get_scores(model, X):
    """Return continuous phishing scores for ROC-AUC/curves."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y, name, stage):
    pred = model.predict(X)
    scores = get_scores(model, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Stage": stage,
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1-score": f1_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC-AUC": roc_auc_score(y, scores),
        "Average Precision": average_precision_score(y, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }, pred, scores


print("Downloading UCI Phishing Websites dataset...")
data = fetch_ucirepo(id=327)
X = data.data.features.copy()
y_original = data.data.targets.iloc[:, 0]
y = (y_original == -1).astype(int)

# Hash identical feature rows. Because of that all duplicates remain in the same fold
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

# Dataset diagnostics
counts = y.value_counts().sort_index()
class_df = pd.DataFrame({
    "Class": ["Legitimate" if i == 0 else "Phishing" for i in counts.index],
    "Count": counts.values,
    "Percentage": counts.values / len(y) * 100,
})
labels_per_group = pd.DataFrame({"group": groups, "target": y.to_numpy()}).groupby("group")["target"].nunique()
dataset_df = pd.DataFrame([{
    "Websites": len(X), "Features": X.shape[1],
    "Missing values": int(X.isna().sum().sum()),
    "Duplicate feature rows": int(X.duplicated().sum()),
    "Unique feature groups": int(pd.Series(groups).nunique()),
    "Conflicting-label groups": int((labels_per_group > 1).sum()),
    "Legitimate count": int((y == 0).sum()),
    "Phishing count": int((y == 1).sum()),
}])
class_df.to_csv(OUT / "class_distribution.csv", index=False)
dataset_df.to_csv(OUT / "dataset_analysis.csv", index=False)
print("\nDataset summary:\n", dataset_df.to_string(index=False))
print("\nClass distribution:\n", class_df.to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(class_df["Class"], class_df["Count"])
plt.title("Class Distribution in the Phishing Websites Dataset")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=300)
plt.close()

# Hold out one group-stratified fold
# It is NOT used during tuning
outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(outer_cv.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train, g_test = groups[train_idx], groups[test_idx]
overlap = len(set(g_train).intersection(set(g_test)))
if overlap:
    raise RuntimeError(f"Group leakage detected: {overlap} shared groups")

split_df = pd.DataFrame([{
    "Training records": len(X_train), "Testing records": len(X_test),
    "Training phishing percentage": y_train.mean() * 100,
    "Testing phishing percentage": y_test.mean() * 100,
    "Shared feature groups": overlap,
    "Method": "StratifiedGroupKFold; first of five outer folds",
}])
split_df.to_csv(OUT / "split_diagnostics.csv", index=False)

# Kept comparable to the previous experiment.
models = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]),
        {"model__C": [0.1, 1.0, 10.0], "model__solver": ["lbfgs"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20, 30],
         "min_samples_split": [2, 10], "min_samples_leaf": [1, 2, 5]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 20],
         "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.8]},
    ),
    "SVM": (
        Pipeline([("scaler", StandardScaler()),
                  ("model", SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE))]),
        {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01]},
    ),
}

# Group-aware CV for the the training partition
inner_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1)
cv_splits = list(inner_cv.split(X_train, y_train, groups=g_train))

all_rows = []
tuning_rows = []
curve_data = {}
matrices = {}

for name, (estimator, param_grid) in models.items():
    print(f"\nTuning {name}...")
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )
    # Warnings (e.g. convergence) are retained as warnings and not hidden
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X_train, y_train)
    for warning in caught:
        print("Warning:", str(warning.message)[:300])

    best_idx = search.best_index_
    tuning_rows.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "CV F1 standard deviation": search.cv_results_["std_test_score"][best_idx],
        "Best parameters": str(search.best_params_),
        "CV folds": N_SPLITS,
    })

    row, pred, scores = evaluate(search.best_estimator_, X_test, y_test, name, "Tuned")
    all_rows.append(row)
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    matrices[name] = matrix
    curve_data[name] = (scores, pred)

    disp = ConfusionMatrixDisplay(matrix, display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} — tuned")
    plt.tight_layout()
    plt.savefig(OUT / f"{name.replace(' ', '_')}_tuned_confusion_matrix.png", dpi=300)
    plt.close()

    # Save full grid results for transparency
    cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_table.to_csv(OUT / f"{name.replace(' ', '_')}_grid_search.csv", index=False)

results_df = pd.DataFrame(all_rows)
tuning_df = pd.DataFrame(tuning_rows)
results_df.to_csv(OUT / "tuned_test_results.csv", index=False)
tuning_df.to_csv(OUT / "best_hyperparameters_and_cv.csv", index=False)

print("\nBest hyperparameters and CV results:\n", tuning_df.round(4).to_string(index=False))
print("\nTuned held-out test results:\n", results_df.round(4).to_string(index=False))

# Tuned model metric comparison
metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "Specificity", "ROC-AUC"]
results_df.set_index("Model")[metric_cols].plot(kind="bar", ylim=(0, 1), figsize=(12, 6))
plt.title("Tuned Comparison of Four Phishing Detection Models")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "tuned_model_comparison.png", dpi=300)
plt.close()

# Combined confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, matrix) in zip(axes.ravel(), matrices.items()):
    ConfusionMatrixDisplay(matrix, display_labels=["Legitimate", "Phishing"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(name)
fig.tight_layout()
fig.savefig(OUT / "all_tuned_confusion_matrices.png", dpi=300)
plt.close(fig)

# ROC and Precision-Recall curves for the four tuned models
plt.figure(figsize=(8, 6))
for name, (scores, _) in curve_data.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Tuned Models")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tuned_roc_curves.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for name, (scores, _) in curve_data.items():
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves — Tuned Models")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tuned_precision_recall_curves.png", dpi=300)
plt.close()

print("\nTuning completed. Results saved in:", OUT.resolve())
print("Important: test results are from one held-out outer fold; CV scores are calculated only on training data.")
