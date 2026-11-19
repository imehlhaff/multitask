"""
Keep the Measuring Hate Speech annotations at one row per (comment, annotator) and split by comment into train/val/test files.
Neither paper aggregates: their multitask model predicts each annotator's item ratings given their severity.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# The top-level measuring-hate-speech.parquet is missing the target_politics_* columns
RAW_PATH = Path(__file__).parent / "raw" / "data" / "train-00000-of-00001.parquet"
OUT_DIR = Path(__file__).parent / "clean"

SEED = 42
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}
N_SCORE_BINS = 10

# Ordinary comments have at most 6 annotations (rows); the 70 reference set comments have 243-815
REFERENCE_MIN_ANNOTATIONS = 100


def assign_splits(comments):
    """Stratified split of comments on hate_speech_score decile x platform; reference set forced into train."""
    rng = np.random.default_rng(SEED)
    split = pd.Series("train", index=comments.index)

    pool = comments[~comments["is_reference"]]
    score_bin = pd.qcut(pool["hate_speech_score"], N_SCORE_BINS, labels=False)
    strata = score_bin.astype(str) + "_" + pool["platform"].astype(str)
    
    # Random position within each stratum, scaled to [0, 1)
    order = pd.Series(rng.permutation(len(pool)), index=pool.index)
    pos = order.groupby(strata).rank(method="first").sub(1) / strata.map(strata.value_counts())

    train_cut = SPLIT_FRACS["train"]
    val_cut = train_cut + SPLIT_FRACS["val"]
    split[pos.index[pos >= train_cut]] = "val"
    split[pos.index[pos >= val_cut]] = "test"
    return split


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(RAW_PATH)

    # hate_speech_score, std_err, text and platform are constant within comment
    comments = df.groupby("comment_id")[["hate_speech_score", "platform"]].first()
    comments["n_annotations"] = df.groupby("comment_id").size()
    comments["is_reference"] = comments["n_annotations"] >= REFERENCE_MIN_ANNOTATIONS
    comments["split"] = assign_splits(comments)

    df = df.merge(comments[["n_annotations", "is_reference", "split"]], left_on="comment_id", right_index=True)
    # Inverse-frequency weights (Kennedy et al. 2020, Fig. 6) so each comment sums to 1, de-emphasising the reference set
    df["weight"] = 1 / df["n_annotations"]

    # Some scores are exactly -1.0, the library's default mask_value: set ModelConfig(mask_value=...) to e.g. -999
    for name, part in df.groupby("split"):
        part.drop(columns="split").to_csv(OUT_DIR / f"hatespeech_{name}.csv", index=False)
