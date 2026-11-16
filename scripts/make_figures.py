"""Generate README-sized evaluation figures from committed outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.machine_learning import predict_tickets  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "docs" / "images"
PALETTE = ["#0d9488", "#7c3aed", "#16a34a", "#f59e0b", "#dc2626", "#2563eb", "#db2777", "#64748b"]


def _save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def make_f1_figure() -> None:
    report = pd.read_csv(PROJECT_ROOT / "outputs" / "classification_report.csv", index_col=0)
    classes = report.index[~report.index.isin(["accuracy", "macro avg", "weighted avg"])]
    values = report.loc[classes, "f1-score"].sort_values()
    plt.figure(figsize=(8, 4.8))
    plt.barh(values.index, values.values, color=PALETTE[: len(values)])
    plt.xlim(0, 1)
    plt.xlabel("F1 score")
    plt.ylabel("Category")
    plt.title("Test F1 score by category")
    for position, value in enumerate(values):
        plt.text(value + 0.01, position, f"{value:.2f}", va="center")
    _save_figure("per_class_f1.png")


def make_coverage_figure() -> None:
    coverage = pd.read_csv(PROJECT_ROOT / "outputs" / "confidence_coverage.csv")
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(
        coverage["confidence_at_least"],
        coverage["share_of_tickets"] * 100,
        marker="o",
        color=PALETTE[0],
        label="Ticket coverage",
    )
    plt.plot(
        coverage["confidence_at_least"],
        coverage["accuracy_on_those"] * 100,
        marker="o",
        color=PALETTE[1],
        label="Accuracy on covered tickets",
    )
    plt.xlabel("Confidence threshold")
    plt.ylabel("Percent")
    plt.ylim(0, 105)
    plt.title("Confidence threshold coverage and accuracy")
    plt.legend()
    _save_figure("confidence_coverage.png")


def make_priority_figure() -> None:
    review = pd.read_csv(PROJECT_ROOT / "outputs" / "priority_review_sample.csv")
    predictions = predict_tickets(review["clean_text"].tolist())
    frame = pd.DataFrame(
        {
            "Reviewer priority": review["reviewer_priority"].str.title(),
            "Rule priority": [item["priority"] for item in predictions],
        }
    )
    order = ["High", "Medium", "Low"]
    table = pd.crosstab(frame["Reviewer priority"], frame["Rule priority"]).reindex(
        index=order, columns=order, fill_value=0
    )
    plt.figure(figsize=(6.8, 5.2))
    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(PALETTE[0], as_cmap=True),
        cbar=False,
    )
    plt.xlabel("Rule-based priority")
    plt.ylabel("Reviewer priority")
    plt.title("Reviewer versus rule-based priority")
    _save_figure("priority_review_agreement.png")


def main() -> None:
    """Generate the three new evaluation figures."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    make_f1_figure()
    make_coverage_figure()
    make_priority_figure()


if __name__ == "__main__":
    main()

