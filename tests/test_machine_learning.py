"""Tests for the category and priority prediction path."""

from __future__ import annotations

import pandas as pd
import pytest

from machine_learning.machine_learning import (
    DATA_PATH,
    clean_text,
    get_scores,
    load_model,
    predict_ticket,
    predict_tickets,
    softmax,
    train_model,
)

REVIEW_PATH = "outputs/priority_review_sample.csv"


def test_priority_review_parity() -> None:
    """Keep the port aligned with the notebook's reviewed examples."""
    frame = pd.read_csv(REVIEW_PATH)
    if frame["reviewer_priority"].notna().sum() == 0:
        pytest.skip("reviewer_priority is not populated in the checked-in sample")
    predictions = predict_tickets(frame["clean_text"].tolist())
    actual = frame["reviewer_priority"].fillna("").str.lower()
    predicted = pd.Series([item["priority"].lower() for item in predictions])
    exact = float((actual == predicted).mean())
    serious = int(
        (
            ((actual == "high") & (predicted == "low"))
            | ((actual == "low") & (predicted == "high"))
        ).sum()
    )
    assert round(exact, 2) == 0.58
    assert serious == 7
    assert int((predicted == "high").sum()) == 11


def test_clean_text_is_idempotent_on_review_sample() -> None:
    frame = pd.read_csv(REVIEW_PATH)
    assert all(clean_text(value) == value for value in frame["clean_text"])


def test_single_and_batch_predictions_match() -> None:
    text = pd.read_csv(REVIEW_PATH).iloc[0]["clean_text"]
    single = predict_ticket(text)
    batch = predict_tickets([text])[0]
    assert single == batch
    bundle = load_model()
    probabilities = softmax(get_scores(bundle["model"], [text]) / bundle["temperature"])[0]
    assert all(0 < value < 1 for value in probabilities)
    assert probabilities.sum() == pytest.approx(1.0)


def test_train_model_on_sample(tmp_path) -> None:
    if not DATA_PATH.exists():
        pytest.skip("raw CSV is missing; full training data is unavailable")
    sample_path = tmp_path / "sample.csv"
    frame = pd.read_csv(DATA_PATH, nrows=2000)
    frame.to_csv(sample_path, index=False)
    output_path = tmp_path / "category_model.joblib"
    bundle = train_model(output_path, sample_path)
    assert output_path.exists()
    assert set(bundle) == {
        "model",
        "temperature",
        "score_method",
        "classes",
        "cleaning_description",
        "stop_words_choice",
        "model_name",
        "probability_rule",
        "versions",
    }
