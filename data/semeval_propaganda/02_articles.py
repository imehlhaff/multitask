"""
Collapse the paragraph-level files written by 01_clean.py to one row per article, keeping each article in its original
train/dev/test split. A technique is 1 for an article if any of its paragraphs has it (logical OR).

The text is the article's paragraphs, in order, joined by newlines. Paragraphs are every non-empty line of the original
article file, so only blank lines are lost. Test has no gold labels, so its technique columns stay blank.
"""
import json
from pathlib import Path

import pandas as pd

CLEAN_DIR = Path(__file__).parent / "clean"
SPLITS = ["train", "dev", "test"]
NON_TECHNIQUE_COLUMNS = ["text", "technique", "article_id", "paragraph_id"]


def load(split):
    return pd.read_csv(CLEAN_DIR / f"semeval_{split}.csv", dtype={"article_id": str})


def to_articles(paragraphs, techniques):
    paragraphs = paragraphs.sort_values(["article_id", "paragraph_id"], kind="stable")
    grouped = paragraphs.groupby("article_id", sort=False)
    articles = pd.DataFrame(
        {
            "text": grouped["text"].agg("\n".join),
            "n_paragraphs": grouped.size(),
        }
    )
    labelled = paragraphs[techniques].notna().all().all()
    if labelled:
        articles[techniques] = grouped[techniques].max().astype(int)
        # JSON list in the official order, like the paragraph files; no technique is "[]"
        articles["technique"] = [json.dumps([t for t in techniques if row[t]]) for _, row in articles.iterrows()]
    return articles.reset_index().reindex(columns=["text", *techniques, "technique", "article_id", "n_paragraphs"])


if __name__ == "__main__":
    paragraphs = {split: load(split) for split in SPLITS}
    # All 23 official technique columns, as in the paragraph files, so the two levels line up
    techniques = [c for c in paragraphs["train"].columns if c not in NON_TECHNIQUE_COLUMNS]

    articles = {}
    for split, df in paragraphs.items():
        articles[split] = to_articles(df, techniques)
        assert articles[split]["article_id"].is_unique
        assert articles[split]["n_paragraphs"].sum() == len(df)
        articles[split].to_csv(CLEAN_DIR / f"semeval_articles_{split}.csv", index=False)
        print(f"{split}: {len(df):,} paragraphs -> {len(articles[split])} articles")

    # Summary: how much OR-ing changes prevalence, and how long the articles are
    pd.set_option("display.width", 200)
    rows = {}
    for t in techniques:
        rows[t] = {
            "train paragraphs %": 100 * paragraphs["train"][t].mean(),
            "train articles %": 100 * articles["train"][t].mean(),
            "dev articles %": 100 * articles["dev"][t].mean(),
            "train articles": int(articles["train"][t].sum()),
            "dev articles": int(articles["dev"][t].sum()),
        }
    prevalence = pd.DataFrame(rows).T.sort_values("train articles", ascending=False)
    print("\nShare of paragraphs vs articles with each technique\n")
    print(prevalence.round(1).to_string())

    print("\nTechniques per article and article length")
    for split in SPLITS:
        a = articles[split]
        words = a["text"].str.split().str.len()
        line = f"  {split:<5} words: median {words.median():.0f}, 95th percentile {words.quantile(0.95):.0f}"
        if split != "test":
            n_techniques = a[techniques].sum(axis=1)
            line += f" | techniques: median {n_techniques.median():.0f}, none {(n_techniques == 0).mean():.0%}"
        print(line)
