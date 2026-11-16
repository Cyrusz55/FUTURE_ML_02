"""Upload the trained category model and its metadata to Hugging Face."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Direct execution starts in scripts/, so add the project root before importing sibling packages.
from apps.model_loader import HF_FILENAME, HF_REPO_ID, HF_REVISION  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "model" / HF_FILENAME
TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "test_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "run_summary.json"


def _read_metrics() -> tuple[dict[str, str], dict[str, object]]:
    with TEST_METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        metrics = next(csv.DictReader(handle))
    with SUMMARY_PATH.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return metrics, summary


def _model_card() -> str:
    metrics, summary = _read_metrics()
    versions = summary["versions"]
    coverage = next(
        row for row in summary["coverage"] if row["confidence_at_least"] == 0.8
    )
    return f"""---
library_name: scikit-learn
pipeline_tag: text-classification
---

# IT ticket category classifier

This artefact predicts one of eight IT help desk categories from free-text ticket
content. It contains a TF-IDF and LinearSVC pipeline with calibrated confidence
scores; transparent priority rules are separate from this artefact.

## Results

- Test accuracy: {float(metrics["test_accuracy"]):.4f}
- Test macro F1: {float(metrics["test_macro_f1"]):.4f}
- Temperature-scaled log loss: {float(metrics["test_log_loss"]):.4f}
- Temperature-scaled calibration error: {float(metrics["expected_calibration_error"]):.4f}
- Confidence >= 0.8 coverage: {float(coverage["share_of_tickets"]):.1%}
- Accuracy at confidence >= 0.8: {float(coverage["accuracy_on_those"]):.1%}
- Calibration temperature: {float(summary["temperature"]):.6f}

## Library versions

{chr(10).join(f"- {name}: {version}" for name, version in versions.items())}

## Limitations

- The data comes from one help desk and may not represent other organisations.
- Priority is a separate rule layer and is not part of this model artefact.
"""


def _tag_exists(api: HfApi) -> bool:
    refs = api.list_repo_refs(repo_id=HF_REPO_ID, repo_type="model")
    return any(tag.name == HF_REVISION for tag in refs.tags)


def main() -> int:
    """Create the public repo, upload artefacts, and create the immutable release tag."""
    load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is missing from .env; upload aborted.", file=sys.stderr)
        return 1
    if not MODEL_PATH.exists():
        print(f"Model artefact is missing: {MODEL_PATH}", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
        if _tag_exists(api):
            print(f"Tag {HF_REVISION} already exists; refusing to move it.", file=sys.stderr)
            return 1
        api.upload_file(
            path_or_fileobj=str(MODEL_PATH),
            path_in_repo=HF_FILENAME,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Upload category classifier artefact",
        )
        api.upload_file(
            path_or_fileobj=_model_card().encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
        api.create_tag(
            repo_id=HF_REPO_ID,
            tag=HF_REVISION,
            repo_type="model",
            tag_message="Release model v1",
        )
    except Exception as exc:
        print(f"Hugging Face upload failed: {exc}", file=sys.stderr)
        return 1
    print(f"Uploaded {HF_FILENAME} and created tag {HF_REVISION} in {HF_REPO_ID}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

