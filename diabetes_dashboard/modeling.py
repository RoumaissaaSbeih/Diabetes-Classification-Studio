from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from .config import (
    DEFAULT_RANDOM_STATE,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    ZERO_AS_MISSING,
    ensure_directories,
)
from .data import build_dataset_summary, build_preview_records, compute_feature_ranges, load_dataset


def replace_invalid_zero_entries(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in ZERO_AS_MISSING:
        cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def _scaled_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("zero_cleanup", FunctionTransformer(replace_invalid_zero_entries, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _tree_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("zero_cleanup", FunctionTransformer(replace_invalid_zero_entries, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }


def benchmark_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict[str, Any]]:
    candidates: dict[str, Pipeline] = {
        "Logistic Regression": _scaled_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=DEFAULT_RANDOM_STATE)
        ),
        "Random Forest": _tree_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=DEFAULT_RANDOM_STATE,
            )
        ),
        "Support Vector Machine": _scaled_pipeline(
            CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", class_weight="balanced", random_state=DEFAULT_RANDOM_STATE),
                ensemble=False,
            )
        ),
    }

    results: list[dict[str, Any]] = []
    for model_name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = _metric_bundle(y_test, predictions, probabilities)
        results.append({"model": model_name, **metrics})

    return sorted(results, key=lambda item: item["roc_auc"], reverse=True)


def train_tuned_knn(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipeline = _scaled_pipeline(KNeighborsClassifier())
    param_grid = {
        "classifier__n_neighbors": list(range(5, 26, 2)),
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc",
        refit=True,
    )
    search.fit(x_train, y_train)
    return search


def build_training_payload() -> dict[str, Any]:
    frame = load_dataset()
    x = frame[FEATURES]
    y = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=y,
    )

    benchmark_results = benchmark_models(x_train, x_test, y_train, y_test)
    search = train_tuned_knn(x_train, y_train)
    final_model = search.best_estimator_

    y_pred = final_model.predict(x_test)
    y_score = final_model.predict_proba(x_test)[:, 1]
    metrics = _metric_bundle(y_test, y_pred, y_score)

    importances = permutation_importance(
        final_model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=DEFAULT_RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    feature_importance = sorted(
        [
            {"feature": feature, "importance": round(float(score), 4)}
            for feature, score in zip(FEATURES, importances.importances_mean)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": build_dataset_summary(frame),
        "feature_ranges": compute_feature_ranges(frame),
        "preview_rows": build_preview_records(frame),
        "benchmark_results": benchmark_results,
        "final_model": {
            "name": "Tuned KNN Classifier",
            "best_params": search.best_params_,
            "cv_best_score": round(float(search.best_score_), 4),
            "test_metrics": metrics,
            "classification_report": class_report,
            "confusion_matrix": matrix.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "feature_importance": feature_importance,
        },
    }

    return {
        "payload": _to_serializable(payload),
        "model": final_model,
    }


def train_and_save_artifacts(force: bool = False) -> dict[str, Any]:
    ensure_directories()

    if MODEL_PATH.exists() and METRICS_PATH.exists() and not force:
        return load_metrics()

    training_output = build_training_payload()
    payload = training_output["payload"]

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    joblib.dump(
        {
            "model": training_output["model"],
            "feature_names": FEATURES,
            "trained_at": payload["generated_at"],
            "model_name": payload["final_model"]["name"],
        },
        MODEL_PATH,
    )

    return payload


def load_metrics() -> dict[str, Any]:
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_bundle() -> dict[str, Any]:
    return joblib.load(MODEL_PATH)


def predict_patient(profile: dict[str, float]) -> dict[str, Any]:
    bundle = load_model_bundle()
    model = bundle["model"]
    frame = pd.DataFrame([profile], columns=bundle["feature_names"])
    probability = float(model.predict_proba(frame)[:, 1][0])
    prediction = int(model.predict(frame)[0])
    return {
        "probability": probability,
        "prediction": prediction,
        "label": "High diabetes risk pattern" if prediction == 1 else "Lower diabetes risk pattern",
    }
