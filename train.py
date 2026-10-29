"""Compare classifiers on the heart disease coursework dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["age", "bp", "chol"]
BINARY = ["sex", "chest.pain"]
FEATURES = NUMERIC + BINARY
SEED = 42


def load_data(path):
    frame = pd.read_csv(path, na_values=["?", "NA", ""])
    expected = set(FEATURES + ["disease"])
    if set(frame.columns) != expected:
        raise ValueError(f"Expected columns: {sorted(expected)}")
    frame["disease"] = frame["disease"].astype("string").str.strip().str.lower()
    if frame["disease"].isna().any() or not frame["disease"].isin(["yes", "no"]).all():
        raise ValueError("Target must contain only yes/no labels without missing values.")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"Infinite values in {column}")
        if frame[column].isna().all():
            raise ValueError(f"No observed values in {column}")
    for column in BINARY:
        if not frame[column].dropna().isin([0, 1]).all():
            raise ValueError(f"Expected binary values in {column}")
    for column in NUMERIC:
        if (frame[column].dropna() <= 0).any():
            raise ValueError(f"Expected positive values in {column}")
    audit = {
        "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "raw_rows": len(frame),
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows_removed": int(frame.duplicated().sum()),
    }
    frame = frame.drop_duplicates().copy()
    conflicting = frame.groupby(FEATURES, dropna=False)["disease"].nunique().gt(1)
    if conflicting.any():
        raise ValueError("Identical feature rows have conflicting labels.")
    audit.update(clean_rows=len(frame), class_counts=frame.disease.value_counts().to_dict())
    return frame[FEATURES], frame["disease"].map({"no": 0, "yes": 1}), audit


def make_pipeline(model):
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", continuous, NUMERIC),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("preprocess", preprocess),
        ("select", SelectKBest(f_classif)),
        ("model", model),
    ])


def candidates(seed):
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed),
            {"model__max_depth": [2, 4, None], "model__min_samples_leaf": [5, 15]},
        ),
    }


def scores(y, predicted, probability):
    return {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }


def tune(model, grid, X, y, seed, jobs):
    search = GridSearchCV(
        make_pipeline(model), {**grid, "select__k": [3, "all"]},
        scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        n_jobs=jobs, refit=True, error_score="raise",
    )
    return search.fit(X, y)


def run(data_path, output, seed=SEED, jobs=1):
    output = Path(output)
    results, figures = output / "results", output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    X, y, audit = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    audit.update(seed=seed, training_rows=len(X_train), test_rows=len(X_test))
    (results / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame({"source_row": X.index, "partition": [
        "train" if index in X_train.index else "test" for index in X.index
    ]}).to_csv(results / "split.csv", index=False)
    rows = []
    outer = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_train, y_train))
    for name, (model, grid) in candidates(seed).items():
        for fold, (train_idx, valid_idx) in enumerate(outer, start=1):
            search = tune(model, grid, X_train.iloc[train_idx], y_train.iloc[train_idx], seed, jobs)
            valid = X_train.iloc[valid_idx]
            row = {"model": name, "fold": fold, **scores(
                y_train.iloc[valid_idx], search.predict(valid), search.predict_proba(valid)[:, 1],
            ), "best_parameters": json.dumps(search.best_params_, sort_keys=True)}
            rows.append(row)
        print(f"Finished nested CV: {name}", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(results / "cv_folds.csv", index=False)
    metrics = list(scores([0, 1], [0, 1], [0., 1.]))
    summary = folds.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(results / "cv_summary.csv")
    winner = summary.index[0]
    model, grid = candidates(seed)[winner]
    final = tune(model, grid, X_train, y_train, seed, jobs)
    predicted, probability = final.predict(X_test), final.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    report = {
        "selected_model": winner,
        "selection_metric": "mean outer-fold ROC-AUC on training partition",
        "best_parameters": final.best_params_,
        "selected_features": final.best_estimator_["preprocess"].get_feature_names_out()[
            final.best_estimator_["select"].get_support()
        ].tolist(),
        "test_metrics": scores(y_test, predicted, probability),
        "baseline_test_metrics": scores(y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]),
        "test_confusion_matrix_order": ["no", "yes"],
    }
    (results / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame({"source_row": X_test.index, "actual": y_test,
                  "predicted": predicted, "probability_yes": probability}).to_csv(
        results / "test_predictions.csv", index=False,
    )
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(summary.index[::-1], summary.roc_auc_mean.iloc[::-1],
            xerr=summary.roc_auc_std.iloc[::-1], color="#246A83", capsize=4)
    ax.set(xlim=(0, 1), xlabel="ROC-AUC (mean ± fold SD)", title="Nested cross-validation")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["No", "Yes"],
                                           ax=axes[0], colorbar=False, cmap="Blues")
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1], name=winner)
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("Held-out test confusion matrix")
    axes[1].set_title("Held-out test ROC curve")
    fig.tight_layout()
    fig.savefig(figures / "test_evaluation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/heart_disease_midterm.csv")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.data, args.output, args.seed, args.jobs)
