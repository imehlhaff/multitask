"""
Train two multitask models on semeval_train and validate them on semeval_dev, both written by 01_clean.py:
one binary head per technique column, and one multilabel head on the "technique" column.
With --write-predictions, also writes each model's dev predictions in the format scorer-subtask-3.py reads.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from multitask import MultiTaskModel, Trainer, TrainingConfig
from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig

pd.set_option("display.max_columns", None)

CLEAN_DIR = Path(__file__).parent / "clean"
RESULTS_DIR = Path(__file__).parent / "results"
BUNDLE_DIR = Path(__file__).parent / "semeval2023task3bundle-v4"

# Tiny BERT for CPU run; swap in a larger encoder for a larger run.
ENCODER = "google/bert_uncased_L-2_H-128_A-2"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42

MULTILABEL_TASK = "technique"
NON_BINARY_COLUMNS = ["text", MULTILABEL_TASK, "article_id", "paragraph_id"]


tokenizer = AutoTokenizer.from_pretrained(ENCODER)


def load(split):
    return pd.read_csv(CLEAN_DIR / f"semeval_{split}.csv", dtype={"article_id": str})


def binary_labels(df, techniques):
    return {t: df[t].to_numpy("float32") for t in techniques}


def multilabel_labels(df, techniques):
    rows = df[MULTILABEL_TASK].map(json.loads)
    return {MULTILABEL_TASK: np.array([[t in row for t in techniques] for row in rows], dtype="float32")}


def make_dataset(df, labels, shuffle=False):
    X = dict(
        tokenizer(df["text"].tolist(), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np")
    )
    ds = tf.data.Dataset.from_tensor_slices((X, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train_and_validate(train, dev, labels_fn, techniques, task_names, task_types, num_classes_per_task):
    """Returns the library's per-task metrics and the [paragraphs, techniques] matrix of dev predictions."""
    tf.keras.utils.set_random_seed(SEED)
    config = ModelConfig(
        task_structure=[[len(task_names)]],  # One flat branch with every head side by side (no hierarchy)
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        encoder_config=EncoderConfig(
            encoder_identifier=ENCODER,
            encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            # The tiny BERT only ships PyTorch weights, so convert them to TF on load
            from_pt=True,
        ),
    )
    train_ds = make_dataset(train, labels_fn(train, techniques), shuffle=True)
    dev_ds = make_dataset(dev, labels_fn(dev, techniques))

    trainer = Trainer(MultiTaskModel(config), TrainingConfig(epochs=EPOCHS, batch_size=BATCH_SIZE))
    trainer.fit(train_ds, dev_ds, num_tasks=len(task_names))
    results = trainer.evaluate(dev_ds)

    # Binary heads each give a 1D array in technique order; the multilabel head gives the whole matrix
    predictions = np.column_stack([results["predictions_dict"][name] for name in task_names])
    return results["task_metrics"], predictions


def write_scorer_file(dev, predictions, techniques, path):
    """One line per paragraph: article_id<TAB>paragraph_id<TAB>technique1,technique2 (empty if none)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for article_id, paragraph_id, row in zip(dev["article_id"], dev["paragraph_id"], predictions):
            names = ",".join(t for t, predicted in zip(techniques, row) if predicted)
            f.write(f"{article_id}\t{paragraph_id}\t{names}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-predictions", action="store_true", help=f"Write dev predictions to {RESULTS_DIR}")
    args = parser.parse_args()

    train, dev = load("train"), load("dev")

    # Columns that never vary in train can't be learned; in English these are 4 all-zero techniques
    columns = [c for c in train.columns if c not in NON_BINARY_COLUMNS]
    techniques = [c for c in columns if train[c].nunique() > 1]
    print(f"Dropped constant columns: {sorted(set(columns) - set(techniques))}")

    n = len(techniques)
    runs = {
        "binary": train_and_validate(train, dev, binary_labels, techniques, techniques, ["binary"] * n, [2] * n),
        "multilabel": train_and_validate(train, dev, multilabel_labels, techniques, [MULTILABEL_TASK], ["multilabel"], [n]),
    }

    gold = multilabel_labels(dev, techniques)[MULTILABEL_TASK]
    comparison = {}
    for name, (task_metrics, predictions) in runs.items():
        print(f"\n{name}: library metrics (weighted F1)")
        print(pd.DataFrame(task_metrics).T)
        # Same averaging as the scorer, so the two models can be compared. Macro is over the kept techniques only.
        comparison[name] = {
            "micro_f1": f1_score(gold, predictions, average="micro", zero_division=1),
            "macro_f1": f1_score(gold, predictions, average="macro", zero_division=1),
        }
    print(f"\nDev F1 over {n} techniques")
    print(pd.DataFrame(comparison).T)

    if args.write_predictions:
        RESULTS_DIR.mkdir(exist_ok=True)
        print("\nScore the predictions with:")
        for name, (_, predictions) in runs.items():
            path = RESULTS_DIR / f"semeval_dev_{name}.txt"
            write_scorer_file(dev, predictions, techniques, path)
            print(
                f"python {BUNDLE_DIR / 'scorers' / 'scorer-subtask-3.py'}"
                f" -g {BUNDLE_DIR / 'data' / 'en' / 'dev-labels-subtask-3.txt'}"
                f" -p {path}"
                f" -f {BUNDLE_DIR / 'scorers' / 'techniques_subtask3.txt'}"
            )


if __name__ == "__main__":
    main()
