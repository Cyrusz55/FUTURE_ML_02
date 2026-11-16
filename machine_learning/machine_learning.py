"""Train and serve the IT ticket category and priority logic."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "category_model.joblib"
DATA_PATH = PROJECT_ROOT / "raw_data" / "all_tickets_processed_improved_v3.csv"
TEXT_COL = "Document"
TARGET_COL = "Topic_group"
RANDOM_STATE = 42
TEST_SIZE = 0.20
HIGH_AT = 4
MEDIUM_AT = 2
RULES_VERSION = 3
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.02), np.log(5.0), 150))

URGENCY_RULES = [
    ("explicit urgency", 3, r"\b(urgent\w*|asap|immediate\w*|emergency|critical|deadline)\b"),
    ("service failure", 3, r"\b(outage|down|crash\w*|fail\w*|freez\w*|frozen|corrupt\w*|dead)\b"),
    (
        "device or feature not working",
        2,
        r"\b(broken|not working|stopped working|no longer working)\b",
    ),
    (
        "security or data risk",
        3,
        (
            r"\b(security|breach\w*|virus|malware|phishing|hack\w*|ransomware|"
            r"leak\w*|stolen|suspicious|data loss)\b"
        ),
    ),
    (
        "blocked from working",
        2,
        r"\b(unable|cannot|blocked|locked|lockout|denied|deny|unavailable|stuck|cant)\b|\bcan t\b",
    ),
    (
        "wide impact",
        2,
        (
            r"\b(everyone|everybody|all users|all staff|whole team|entire team|"
            r"entire office|multiple users|company wide|team wide)\b"
        ),
    ),
    ("sender marked high importance", 1, r"\bimportance high\b"),
    ("degraded service", 1, r"\b(slow\w*|lag\w*|intermittent\w*|delay\w*|sometimes|error\w*)\b"),
    (
        "routine request",
        -2,
        (
            r"\b(how to|how do|question|inquiry|new starter|upgrade|quote|would like|"
            r"wondering|recommend\w*)\b"
        ),
    ),
    ("scheduled or planned", -3, r"\b(scheduled|planned)\b"),
    ("maintenance notice", -4, r"\bmaintenance\b"),
]
COMPILED_RULES = [(name, points, re.compile(pattern)) for name, points, pattern in URGENCY_RULES]
CATEGORY_WEIGHT_HINTS = {
    "access": 1,
    "administrative": 1,
    "hardware": 1,
    "storage": 1,
    "purchase": -1,
    "miscellaneous": -1,
}

IMBALANCE_RATIO: float | None = None


def clean_text(value: Any) -> str:
    """Apply the notebook's text normalization."""
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_X_y(data_path: Path = DATA_PATH) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Load, clean, and encode the dataset used by the notebook."""
    global IMBALANCE_RATIO
    frame = pd.read_csv(data_path)[[TEXT_COL, TARGET_COL]].dropna().copy()
    frame[TEXT_COL] = frame[TEXT_COL].astype(str).map(clean_text)
    frame[TARGET_COL] = frame[TARGET_COL].astype(str).str.strip()
    frame = frame[frame[TEXT_COL] != ""].reset_index(drop=True)
    counts = frame[TARGET_COL].value_counts()
    IMBALANCE_RATIO = float(counts.max() / counts.min())
    encoder = LabelEncoder().fit(frame[TARGET_COL])
    return frame[TEXT_COL], encoder.transform(frame[TARGET_COL]), list(encoder.classes_)


def build_pipeline() -> Pipeline:
    """Build the notebook's final TF-IDF and LinearSVC pipeline."""
    # Resampling is unnecessary because the notebook's balanced class weights handle imbalance.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LinearSVC(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_scores(model: Any, texts: list[str] | pd.Series) -> np.ndarray:
    """Return decision scores, or log probabilities for probabilistic estimators."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts))
    return np.log(np.clip(model.predict_proba(texts), 1e-9, 1.0))


def softmax(scores: np.ndarray) -> np.ndarray:
    """Calculate row-wise numerically stable softmax probabilities."""
    values = np.asarray(scores)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    shifted = values - values.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit temperature using the notebook's fixed log-loss grid."""
    losses = []
    for temperature in TEMPERATURE_GRID:
        probabilities = softmax(scores / temperature)
        chosen = probabilities[np.arange(len(labels)), labels]
        losses.append(float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))))
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def train_model(output_path: Path, data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Fit the notebook model and save its bundle format."""
    features, labels, classes = get_X_y(data_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, _ = next(splitter.split(features, labels, groups=features))
    x_train = features.iloc[train_idx]
    y_train = labels[train_idx]
    model = build_pipeline().fit(x_train, y_train)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = cross_val_predict(
        clone(build_pipeline()),
        x_train,
        y_train,
        groups=x_train,
        cv=cv,
        method="decision_function",
        n_jobs=1,
    )
    temperature = fit_temperature(oof_scores, y_train)
    bundle = {
        "model": model,
        "temperature": temperature,
        "score_method": "decision_function",
        "classes": classes,
        "cleaning_description": (
            "Lowercase, remove web links and tags, replace punctuation with spaces, "
            "collapse whitespace. Stop words are removed inside the vectorizer."
        ),
        "stop_words_choice": "standard English stop words",
        "model_name": "LinearSVC",
        "probability_rule": (
            "softmax(scores / temperature), scores from decision_function "
            "(log of probabilities when predict_proba is used)"
        ),
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load and validate a model bundle."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "temperature", "classes"}.issubset(bundle):
        raise ValueError("Model artefact is not a supported category model bundle.")
    return bundle


def _category_weight(category_name: str) -> int:
    name = str(category_name).lower()
    for hint, weight in CATEGORY_WEIGHT_HINTS.items():
        if hint in name:
            return weight
    return 0


def _level_from_points(points: int) -> str:
    if points >= HIGH_AT:
        return "High"
    if points >= MEDIUM_AT:
        return "Medium"
    return "Low"


def assign_priority(clean: str, category_name: str) -> tuple[str, int, list[str], list[str]]:
    """Apply the notebook's transparent priority rules."""
    matched_rules: list[str] = []
    matched_terms: list[str] = []
    points = 0
    for name, rule_points, pattern in COMPILED_RULES:
        found = pattern.search(clean)
        if found:
            matched_rules.append(name)
            matched_terms.append(found.group(0))
            points += rule_points
    points += _category_weight(category_name)
    return _level_from_points(points), points, matched_rules, matched_terms


def predict_tickets(texts: list[str]) -> list[dict[str, Any]]:
    """Predict category confidence and transparent priority for each ticket."""
    bundle = load_model()
    cleaned = [clean_text(text) for text in texts]
    if any(not text for text in cleaned):
        raise ValueError("The ticket text is empty after cleaning.")
    probabilities = softmax(get_scores(bundle["model"], cleaned) / float(bundle["temperature"]))
    classes = list(bundle["classes"])
    results = []
    for clean, row in zip(cleaned, probabilities):
        index = int(np.argmax(row))
        category = classes[index]
        priority, points, rules, _ = assign_priority(clean, category)
        top_indices = np.argsort(row)[::-1][:3]
        results.append(
            {
                "category": category,
                "category_confidence": round(float(row[index]), 3),
                "top_categories": [
                    {"category": classes[int(i)], "probability": round(float(row[i]), 3)}
                    for i in top_indices
                ],
                "priority": priority,
                "priority_points": points,
                "priority_rules_matched": rules,
                "priority_method": "rule-based; not a trained model",
            }
        )
    return results


def predict_ticket(text: str) -> dict[str, Any]:
    """Predict one ticket through the batch implementation."""
    return predict_tickets([text])[0]

