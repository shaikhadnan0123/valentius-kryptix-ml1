import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = Path("data/Telco-Customer-Churn.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    # customerID is an identifier, not a feature.
    df = df.drop(columns=["customerID"])

    # TotalCharges is read as object because 11 brand-new customers (tenure = 0)
    # have a blank string instead of a number.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())

    # Those rows are genuinely "no charges yet", so 0 is the honest imputation.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target -> binary
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print(f"Rows: {len(df)}  |  Features: {df.shape[1] - 1}")
    print(f"Missing TotalCharges values imputed with 0: {n_missing}")
    print(f"Class balance -> No churn: {(df.Churn == 0).mean():.1%}, "
          f"Churn: {(df.Churn == 1).mean():.1%}\n")
    return df


# ----------------------------------------------------------------------------
# 2. Identical split + identical preprocessing
# ----------------------------------------------------------------------------
def build_preprocessors(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)

    # For KNN / SVM / Logistic Regression / Naive Bayes: scale the numerics.
    scaled = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", ohe, categorical)]
    )
    # For Decision Tree / Random Forest: splits are scale-invariant, so leave
    # the numerics alone. Same encoding, same columns, same rows.
    unscaled = ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="first",
                               sparse_output=False), categorical)]
    )
    return scaled, unscaled, numeric, categorical


def get_models(scaled, unscaled):
    """Five required algorithms + Random Forest as an ensemble reference point."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", scaled),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("prep", scaled),
            ("clf", KNeighborsClassifier(n_neighbors=25)),
        ]),
        "Support Vector Machine": Pipeline([
            ("prep", scaled),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "Naive Bayes": Pipeline([
            ("prep", scaled),
            ("clf", GaussianNB()),
        ]),
        "Decision Tree": Pipeline([
            ("prep", unscaled),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", unscaled),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


# ----------------------------------------------------------------------------
# 3. Train + evaluate
# ----------------------------------------------------------------------------
def evaluate(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, roc_data, reports = [], {}, {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "CV F1 (train, 5-fold)": cv_f1.mean(),
            "CV F1 std": cv_f1.std(),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_proba))
        reports[name] = classification_report(y_test, y_pred,
                                              target_names=["No churn", "Churn"])
        print(f"trained: {name}")

    results = (pd.DataFrame(rows)
               .sort_values("F1", ascending=False)
               .reset_index(drop=True))
    return results, roc_data, reports


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------
def plot_comparison(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    long = results.melt(id_vars="Model", value_vars=metrics,
                        var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Model comparison — Telco Customer Churn (identical split & preprocessing)")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ranked = results.sort_values("F1")
    sns.barplot(data=ranked, y="Model", x="F1", ax=ax, color="#4C72B0")
    for i, v in enumerate(ranked["F1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_title("Ranking by F1 score (churn class)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_ranking.png", dpi=150)
    plt.close(fig)


def plot_roc(roc_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the shared test set")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(models, X_test, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, pipe) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_test, y_test, ax=ax, colorbar=False,
            display_labels=["No churn", "Churn"], cmap="Blues"
        )
        ax.set_title(name, fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices (shared test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"(stratified, random_state={RANDOM_STATE})\n")

    scaled, unscaled, numeric, categorical = build_preprocessors(X)
    print(f"Numeric features ({len(numeric)}): {numeric}")
    print(f"Categorical features ({len(categorical)}): {categorical}\n")

    models = get_models(scaled, unscaled)
    results, roc_data, reports = evaluate(models, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 90)
    print("COMPARISON TABLE (ranked by F1 on the churn class)")
    print("=" * 90)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    results.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    with open(RESULTS_DIR / "classification_reports.txt", "w") as fh:
        for name, rep in reports.items():
            fh.write(f"{'=' * 60}\n{name}\n{'=' * 60}\n{rep}\n")
    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump({
            "dataset": "Telco Customer Churn (7043 rows, 19 features)",
            "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                      "stratified": True},
            "ranked_by": "F1 (churn class)",
            "results": results.to_dict(orient="records"),
        }, fh, indent=2)

    plot_comparison(results)
    plot_roc(roc_data)
    plot_confusion(models, X_test, y_test)
    print(f"\nSaved tables and charts to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
