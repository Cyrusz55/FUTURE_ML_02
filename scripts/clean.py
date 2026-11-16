"""Create the clean ticket CSV used by downstream database loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.machine_learning import clean_text  # noqa: E402

SOURCE_PATH = PROJECT_ROOT / "raw_data" / "all_tickets_processed_improved_v3.csv"
OUTPUT_PATH = PROJECT_ROOT / "raw_data" / "clean" / "tickets_clean.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean source rows using the same text function as model inference."""
    cleaned = df[["Document", "Topic_group"]].dropna().reset_index(drop=True).copy()
    result = pd.DataFrame(
        {
            "ticket_id": range(len(cleaned)),
            "text": cleaned["Document"].astype(str),
            "clean_text": cleaned["Document"].map(clean_text),
            "label": cleaned["Topic_group"].astype(str).str.strip(),
        }
    )
    return result[result["clean_text"] != ""].reset_index(drop=True)


def main() -> None:
    """Read the raw CSV and write its cleaned representation."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_data(pd.read_csv(SOURCE_PATH))
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, test_indices = next(
        split.split(cleaned["clean_text"], groups=cleaned["clean_text"])
    )
    # The notebook's review IDs are positions in its held-out test frame, so keep that order first.
    ordered = pd.concat(
        [cleaned.iloc[test_indices], cleaned.iloc[train_indices]],
        ignore_index=True,
    )
    ordered["ticket_id"] = range(len(ordered))
    ordered.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

