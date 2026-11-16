"""Opt-in integration test for the public Hugging Face model artefact."""

from __future__ import annotations

import pytest
from huggingface_hub import hf_hub_download

from apps.model_loader import HF_FILENAME, HF_REPO_ID, HF_REVISION
from machine_learning.machine_learning import (
    MODEL_PATH,
    assign_priority,
    clean_text,
    get_scores,
    load_model,
    predict_ticket,
    softmax,
)


@pytest.mark.hf
def test_hugging_face_model_matches_local(tmp_path) -> None:
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        revision=HF_REVISION,
        local_dir=tmp_path,
    )
    downloaded = load_model(path=downloaded_path)
    text = "My laptop is broken and crashes urgently."
    local_result = predict_ticket(text)
    clean = clean_text(text)
    probabilities = softmax(get_scores(downloaded["model"], [clean]) / downloaded["temperature"])[0]
    index = int(probabilities.argmax())
    priority, points, rules, _ = assign_priority(clean, downloaded["classes"][index])
    downloaded_result = {
        "category": downloaded["classes"][index],
        "category_confidence": round(float(probabilities[index]), 3),
        "top_categories": [
            {
                "category": downloaded["classes"][int(candidate)],
                "probability": round(float(probabilities[candidate]), 3),
            }
            for candidate in probabilities.argsort()[::-1][:3]
        ],
        "priority": priority,
        "priority_points": points,
        "priority_rules_matched": rules,
        "priority_method": "rule-based; not a trained model",
    }
    assert downloaded_result == local_result
    assert MODEL_PATH.exists()
