"""
Reshape the English SemEval 2023 Task 3 subtask 3 data to one row per paragraph, with one binary column per technique,
and write one file per split. Paragraphs with no technique are kept.
"""
import json
from pathlib import Path

import pandas as pd

BUNDLE_DIR = Path(__file__).parent / "semeval2023task3bundle-v4"
EN_DIR = BUNDLE_DIR / "data" / "en"
TECHNIQUES_PATH = BUNDLE_DIR / "scorers" / "techniques_subtask3.txt"
OUT_DIR = Path(__file__).parent / "clean"

SPLITS = ["train", "dev", "test"]
# The bundle has no gold labels for test
LABELLED_SPLITS = ["train", "dev"]


def read_rows(path):
    """Read a tab-separated file. Not pd.read_csv: article text has unbalanced quotes that must stay as written."""
    with open(path, encoding="utf-8", newline="") as f:
        return [line.split("\t") for line in f.read().split("\n") if line]


def load_paragraphs(split):
    # Merged templates are only safe for English: other languages' train files have rows glued together
    df = pd.DataFrame(read_rows(EN_DIR / f"{split}-labels-subtask-3.template"), columns=["article_id", "paragraph_id", "text"])
    # 16 test articles use CRLF line endings
    df["text"] = df["text"].str.split().str.join(" ")
    return df


def load_labels(split):
    """Map (article_id, paragraph_id) to that paragraph's techniques. Rows with no technique have an empty last column."""
    return {
        (row[0], row[1]): [technique for technique in (row[2] if len(row) > 2 else "").split(",") if technique]
        for row in read_rows(EN_DIR / f"{split}-labels-subtask-3.txt")
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    labels = {split: load_labels(split) for split in LABELLED_SPLITS}

    # 4 of the 23 official techniques never occur in English, so they get no column
    official = TECHNIQUES_PATH.read_text(encoding="utf-8").split()
    observed = {technique for by_paragraph in labels.values() for row in by_paragraph.values() for technique in row}
    assert observed <= set(official)
    techniques = official # Filter for a subset, e.g. not all 0

    for split in SPLITS:
        df = load_paragraphs(split)

        if split in labels:
            keys = list(zip(df["article_id"], df["paragraph_id"]))
            assert set(keys) == set(labels[split])
            rows = [labels[split][key] for key in keys]
            for technique in techniques:
                df[technique] = [int(technique in row) for row in rows]
            # JSON list so the column can be parsed back with json.loads; no technique is "[]"
            df["technique"] = [json.dumps(row) for row in rows]

        # Test has none of the label columns, so reindex leaves them blank: its labels are unknown, not all 0
        df = df.reindex(columns=["text", *techniques, "technique", "article_id", "paragraph_id"])
        df.to_csv(OUT_DIR / f"semeval_{split}.csv", index=False)
