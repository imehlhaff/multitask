"""
Split clean/cap_manifestos.csv into train, val and test by document, so no manifesto or platform has sentences in two
splits. Sentences from one document share topic mix, wording and party style, so splitting by sentence would leak.

Documents range from 11 to 2,774 sentences, so the split targets SPLIT_FRACS of each country's sentences rather than of
its documents. Within each country it shuffles the documents and gives each one to whichever split is furthest below
its target. Splitting within countries keeps all three countries in every split. A shuffle is accepted only if every
major topic has at least MIN_TOPIC_ROWS sentences in val and in test; otherwise it tries the next seed. The result is
deterministic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path(__file__).parent / "clean"

SEED = 42
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}
MIN_TOPIC_ROWS = 30
MAX_TRIES = 1000


def assign_documents(df, seed):
    """Return {doc_id: split}, filling each country's splits towards SPLIT_FRACS of its sentences."""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("doc_id").size()
    country = df.groupby("doc_id")["country"].first()
    assignment = {}
    for c in sorted(country.unique()):
        docs = sizes[country == c]
        docs = docs.iloc[rng.permutation(len(docs))]
        filled = dict.fromkeys(SPLIT_FRACS, 0)
        for doc_id, n in docs.items():
            # The split furthest below its target share of this country's sentences so far, counting this document
            split = min(SPLIT_FRACS, key=lambda s: (filled[s] + n) / docs.sum() - SPLIT_FRACS[s])
            assignment[doc_id] = split
            filled[split] += n
    return assignment


def topics_covered(df, split_of):
    """True if every major topic has at least MIN_TOPIC_ROWS sentences in val and in test."""
    counts = pd.crosstab(df["doc_id"].map(split_of), df["majortopic"]).reindex(["val", "test"], fill_value=0)
    return counts.to_numpy().min() >= MIN_TOPIC_ROWS


def split_documents(df):
    for seed in range(SEED, SEED + MAX_TRIES):
        split_of = assign_documents(df, seed)
        if topics_covered(df, split_of):
            return df["doc_id"].map(split_of), seed
    raise RuntimeError(f"No shuffle in {MAX_TRIES} tries gives every major topic {MIN_TOPIC_ROWS} val and test rows")


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_DIR / "cap_manifestos.csv")
    df["split"], seed = split_documents(df)
    assert (df.groupby("doc_id")["split"].nunique() == 1).all(), "A document is in more than one split"

    for name in SPLIT_FRACS:
        df[df["split"] == name].drop(columns="split").to_csv(CLEAN_DIR / f"cap_{name}.csv", index=False)

    # Report
    print(f"Seed {seed}\n")
    summary = df.groupby("split").agg(documents=("doc_id", "nunique"), sentences=("text", "size")).loc[list(SPLIT_FRACS)]
    summary["share"] = (summary["sentences"] / len(df)).round(3)
    print(summary.to_string(), "\n")
    print("Share of each country's sentences")
    print(pd.crosstab(df["country"], df["split"], normalize="index")[list(SPLIT_FRACS)].round(3).to_string(), "\n")
    topic_counts = pd.crosstab(df["majortopic"], df["split"])[list(SPLIT_FRACS)]
    print(f"Fewest sentences for any major topic: {topic_counts.min().to_dict()}")
    train_subtopics = set(df.loc[df["split"] == "train", "subtopic"])
    for name in ["val", "test"]:
        unseen = set(df.loc[df["split"] == name, "subtopic"]) - train_subtopics
        if unseen:
            rows = int((df["split"].eq(name) & df["subtopic"].isin(unseen)).sum())
            print(f"Subtopics in {name} but not train: {sorted(unseen)} ({rows} sentences)")
