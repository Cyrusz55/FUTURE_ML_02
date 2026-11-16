"""Load and cache the local or pinned Hugging Face model artefact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import sklearn
from huggingface_hub import hf_hub_download

from machine_learning.machine_learning import MODEL_PATH, load_model

LOGGER = logging.getLogger(__name__)
HF_REPO_ID = "cyrusnx/it_ticket_classifier_linearsvc"
HF_FILENAME = "category_model.joblib"
HF_REVISION = "v1"

_model_bundle: dict[str, Any] | None = None


def get_model() -> dict[str, Any]:
    """Return the cached model, downloading it only when the local artefact is absent."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    model_path = MODEL_PATH
    if not model_path.exists():
        model_path = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                revision=HF_REVISION,
                local_dir=MODEL_PATH.parent,
            )
        )
    _model_bundle = load_model(model_path)
    expected_version = _model_bundle.get("versions", {}).get("scikit_learn")
    if expected_version and expected_version != sklearn.__version__:
        LOGGER.warning(
            "Model was trained with scikit-learn %s but the installed version is %s.",
            expected_version,
            sklearn.__version__,
        )
    return _model_bundle


def model_is_loaded() -> bool:
    """Return whether the bundle has been warmed into the process cache."""
    return _model_bundle is not None

