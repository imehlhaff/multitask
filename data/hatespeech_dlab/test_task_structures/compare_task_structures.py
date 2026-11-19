"""
Compare four task structures for predicting the Measuring Hate Speech annotations (Sachdeva et al. 2022) from a
comment's text. There are 11 tasks: the 10 ordinal survey items, each one annotator's rating of the comment and treated
as multiclass as in Kennedy et al. (2020), plus the comment's continuous hate_speech_score as a regression. Every model
has the same encoder and shared layers, but the task_structure above them differs:

  single        11 separate models, one head each. Nothing is shared between tasks, so this is the baseline with no
                multitask learning.
  flat          One model, every task on its own branch off the shared layers: [[11]].
  three_trunks  One model with a trunk per group on Kennedy et al.'s severity scale, each splitting into one branch per
                task: [[3], [7, 2, 2]]. Hostility: sentiment, respect, insult, humiliate, status, attack_defend,
                dehumanize. Violence: violence, genocide. Overall: hatespeech, hate_speech_score.
  two_trunks    As three_trunks with hostility and violence merged, so one trunk for the nine component items and one
                for the overall judgement: [[2], [9, 2]].

Data: one row per (comment, annotator), split by comment in ../01_clean.py. Early stopping uses val, and test is the
evaluation split. The model sees only the text, so it predicts the same ratings for every annotator of a comment.
Annotators disagree (the standard deviation of a comment's ratings on an item averages 0.4-0.9 points), so no model can
match every row. The 70 reference comments were rated by 243-815 annotators each and make up 37% of train rows. The
library's training step ignores sample weights, so the `weight` column can't down-weight them; instead each is cut to
REFERENCE_MAX_ANNOTATIONS randomly chosen annotations, the most any other comment has.

The script has two steps:

  train      Train one model and save its predictions on val and test. --task is a run name or its index (`tasks`
             lists them: 0-10 single tasks, then flat, three_trunks, two_trunks), so a SLURM array can run all 14 jobs.
             Without --task it reads SLURM_ARRAY_TASK_ID.
  aggregate  Score every condition whose runs have all finished, on val and test. Takes seconds on CPU.

Examples:
  python compare_task_structures.py tasks
  python compare_task_structures.py train --task insult
  python compare_task_structures.py train --task three_trunks
  python compare_task_structures.py aggregate
"""
import argparse
import os

# Make tf.keras mean tf-keras (Keras 2), the Keras that transformers builds its TF encoders with. Under Keras 3 the
# model doesn't track the encoder's weights, so the encoder never trains. Must be set before TensorFlow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
CLEAN_DIR = HERE.parent / "clean"
RESULTS_DIR = HERE / "results"

# Kennedy et al. (2020, Fig. 1) order the items from mildest to most severe; these are the groups along that scale
GROUPS = {
    "hostility": ["sentiment", "respect", "insult", "humiliate", "status", "attack_defend", "dehumanize"],
    "violence": ["violence", "genocide"],
    "overall": ["hatespeech", "hate_speech_score"],
}
TASKS = [t for ts in GROUPS.values() for t in ts]
SCORE = "hate_speech_score"
ITEMS = [t for t in TASKS if t != SCORE]
# Ratings run 0-4, except hatespeech (0-2)
NUM_CLASSES = {item: 3 if item == "hatespeech" else 5 for item in ITEMS}

JOINT_CONDITIONS = ["flat", "three_trunks", "two_trunks"]
CONDITIONS = ["single", *JOINT_CONDITIONS]

# Ordinary comments have at most 6 annotations
REFERENCE_MAX_ANNOTATIONS = 6

# Fixed, so every run and training seed sees the same rows
SUBSAMPLE_SEED = 0

# Some hate_speech_score values are exactly -1.0, the library's default mask_value
MASK_VALUE = -999.0

DEFAULTS = {
    "encoder": "roberta-base",
    "max_len": 128,  # 95% of comments have at most 75 words
    "batch_size": 32,
    "epochs": 10,  # The upper limit: early stopping on validation loss (patience 3) usually stops sooner
    "learning_rate": 2e-5,  # The library default of 3e-4 is too high to fine-tune a full transformer
    "seed": 42,
}


# Data


def load(split):
    return pd.read_csv(
        CLEAN_DIR / f"hatespeech_{split}.csv", usecols=["comment_id", "annotator_id", "text", "is_reference", *TASKS]
    )


def cap_reference_comments(train):
    """Keep REFERENCE_MAX_ANNOTATIONS random annotations of each reference comment, in the original row order."""
    reference = train[train["is_reference"]]
    kept = reference.groupby("comment_id").sample(n=REFERENCE_MAX_ANNOTATIONS, random_state=SUBSAMPLE_SEED)
    return pd.concat([train[~train["is_reference"]], kept]).sort_index().reset_index(drop=True)


def row_keys(df):
    # A string array, not pandas' object array, so np.load can read it back without pickle
    return (df["comment_id"].astype(str) + "_" + df["annotator_id"].astype(str)).to_numpy(dtype=str)


def load_splits():
    """Return the train rows to fit on (reference comments cut down), val and test."""
    return cap_reference_comments(load("train")), load("val"), load("test")


# Task structures


def task_type(task):
    return "regression" if task == SCORE else "multiclass"


def model_spec(run):
    """Return (task_structure, tasks) for one run, with tasks in the model's output order."""
    if run in TASKS:
        return [[1]], [run]
    if run == "flat":
        return [[len(TASKS)]], TASKS
    if run == "three_trunks":
        # Level 0: one branch per group. Level 1: each group branch splits into one branch per task.
        return [[len(GROUPS)], [len(ts) for ts in GROUPS.values()]], TASKS
    if run == "two_trunks":
        components = GROUPS["hostility"] + GROUPS["violence"]
        return [[2], [len(components), len(GROUPS["overall"])]], components + GROUPS["overall"]
    raise ValueError(run)


def all_runs():
    return [*TASKS, *JOINT_CONDITIONS]


def condition_of(run):
    return run if run in JOINT_CONDITIONS else "single"


def resolve_run(task):
    runs = all_runs()
    if task is None:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            raise SystemExit("Pass --task, or run inside a SLURM array job")
    if task.isdigit():
        if int(task) >= len(runs):
            raise SystemExit(f"Task index {task} is out of range: there are {len(runs)} runs (0-{len(runs) - 1})")
        return runs[int(task)]
    if task not in runs:
        raise SystemExit(f"Unknown task {task!r}. Choose from: {', '.join(runs)}")
    return task


def run_dir(args):
    name = args.run_name or f"{args.encoder.replace('/', '--')}_seed{args.seed}"
    return RESULTS_DIR / name


# Train


def train(args):
    # TensorFlow is only needed here, so `aggregate` and `tasks` run without it
    import tensorflow as tf
    from transformers import AutoTokenizer

    from multitask import MultiTaskModel, Trainer, TrainingConfig
    from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig

    fit, val, test = load_splits()
    run = resolve_run(args.task)
    out_dir = run_dir(args)
    out_path = out_dir / f"{run}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to retrain")
        return

    task_structure, tasks = model_spec(run)
    print(f"Run: {run} (condition {condition_of(run)}), task_structure {task_structure}, tasks: {', '.join(tasks)}")
    print(f"Rows: {len(fit)} train ({fit['comment_id'].nunique()} comments), {len(val)} validation, {len(test)} test")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def make_dataset(df, shuffle=False):
        X = dict(
            tokenizer(
                df["text"].tolist(), padding="max_length", truncation=True, max_length=args.max_len, return_tensors="np"
            )
        )
        y = {t: df[t].to_numpy("float32" if t == SCORE else "int32") for t in tasks}
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(len(df), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    tf.keras.utils.set_random_seed(args.seed)
    config = ModelConfig(
        task_structure=task_structure,
        task_names=tasks,
        task_types=[task_type(t) for t in tasks],
        num_classes_per_task=[1 if t == SCORE else NUM_CLASSES[t] for t in tasks],
        mask_value=MASK_VALUE,
        encoder_config=EncoderConfig(
            encoder_identifier=args.encoder,
            encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            # Every HF model ships PyTorch weights, but not all ship TF weights (e.g. the tiny BERTs)
            from_pt=True,
        ),
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_sequence_length=args.max_len,
        # The package's warmup schedule is written for Keras 3 and crashes the tf-keras optimizer
        use_warmup=False,
    )
    fit_ds, val_ds, test_ds = make_dataset(fit, shuffle=True), make_dataset(val), make_dataset(test)

    class Model(MultiTaskModel):
        """MultiTaskModel's train_step and test_step return the latest batch's loss, and tf-keras logs that as the
        epoch's loss, so val_loss (which early stopping watches) would be the last validation batch alone. Both
        steps update a metric named "loss" if the model has one, so give it a running mean that Keras resets each
        epoch."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_mean = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_mean]

    model = Model(config)
    model(next(iter(fit_ds))[0])  # Build it, so its weights exist
    tracked = {id(v) for v in model.trainable_variables}
    if not all(id(v) in tracked for v in model.encoder.trainable_weights):
        raise SystemExit("The encoder's weights aren't among the model's trainable variables, so it wouldn't train")
    trainer = Trainer(model, training_config)
    start = time.time()
    history, _ = trainer.fit(fit_ds, val_ds, num_tasks=len(tasks))
    seconds = time.time() - start

    def predictions(ds, split):
        """Class probabilities [rows, classes] per item, and the predicted score [rows, 1]."""
        outputs = trainer.model.predict(ds, verbose=0)
        return {f"{split}_{t}": np.asarray(outputs[t]) for t in tasks}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        run=run,
        task_structure=str(task_structure),
        tasks=np.array(tasks),
        **predictions(val_ds, "val"),
        **predictions(test_ds, "test"),
        val_keys=row_keys(val),
        test_keys=row_keys(test),
        epochs_trained=len(history.history["loss"]),
        best_val_loss=min(history.history["val_loss"]),
        train_seconds=seconds,
        encoder=args.encoder,
        seed=args.seed,
    )
    print(f"Trained {len(history.history['loss'])} epochs in {seconds / 60:.1f} min. Wrote {out_path}")


# Aggregate


def load_condition_predictions(directory, splits):
    """Return {condition: {split: {task: predictions}}} for every condition whose runs have all finished, plus a table
    of training statistics per run."""
    runs_by_condition = {"single": TASKS, **{c: [c] for c in JOINT_CONDITIONS}}
    predictions, training, missing = {}, {}, []
    for condition, runs in runs_by_condition.items():
        absent = [r for r in runs if not (directory / f"{r}.npz").exists()]
        if absent:
            missing += absent
            continue
        merged = {split: {} for split in splits}
        for name in runs:
            run = np.load(directory / f"{name}.npz")
            for split, df in splits.items():
                # Guard against mixing runs from different data
                assert (run[f"{split}_keys"] == row_keys(df)).all(), f"{name}: {split} rows differ from this data"
                for t in run["tasks"]:
                    merged[split][str(t)] = run[f"{split}_{t}"]
            training[name] = {
                "condition": condition,
                "epochs": int(run["epochs_trained"]),
                "best_val_loss": float(run["best_val_loss"]),
                "minutes": float(run["train_seconds"]) / 60,
            }
        assert sorted(merged["test"]) == sorted(TASKS), f"{condition}: runs don't cover the tasks exactly once"
        predictions[condition] = merged

    if missing:
        print(f"Not scored yet, {len(missing)} runs missing in {directory}: {', '.join(missing)}\n")
    if not predictions:
        raise SystemExit("No condition has all its runs")
    return predictions, pd.DataFrame(training).T


def baseline_predictions(fit, n):
    """Predict train's most common rating for every item and train's mean score, the floor any model should beat."""
    out = {}
    for item in ITEMS:
        probs = np.zeros((n, NUM_CLASSES[item]))
        probs[:, fit[item].astype(int).mode()[0]] = 1
        out[item] = probs
    out[SCORE] = np.full((n, 1), fit[SCORE].mean())
    return out


def score_item(y, probs, num_classes):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(num_classes), zero_division=0),
        # The ratings are ordinal, so how far off a wrong rating is matters too
        "mae": np.abs(pred - y).mean(),
    }


def score_regression(df, pred):
    """Scored once per comment, since hate_speech_score is a property of the comment, not the annotator."""
    comments = pd.DataFrame({"comment_id": df["comment_id"], "y": df[SCORE], "pred": pred.ravel()})
    comments = comments.groupby("comment_id").mean()
    error = comments["pred"] - comments["y"]
    return {
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(error.abs().mean()),
        "pearson_r": float(comments["pred"].corr(comments["y"])) if comments["pred"].std() > 0 else np.nan,
    }


def aggregate(args):
    fit, val, test = load_splits()
    splits = {"val": val, "test": test}
    directory = run_dir(args)
    predictions, training = load_condition_predictions(directory, splits)
    predictions = {"baseline": {s: baseline_predictions(fit, len(df)) for s, df in splits.items()}, **predictions}

    summary, per_task = {}, []
    for condition, by_split in predictions.items():
        for split, df in splits.items():
            rows = []
            for item in ITEMS:
                scores = score_item(df[item].astype(int).to_numpy(), by_split[split][item], NUM_CLASSES[item])
                rows.append({"task": item, "condition": condition, "split": split, **scores})
            per_task += rows
            regression = score_regression(df, by_split[split][SCORE])
            per_task.append({"task": SCORE, "condition": condition, "split": split, **regression})
            items = pd.DataFrame(rows)
            summary[(condition, split)] = {
                "items accuracy (mean)": items["accuracy"].mean(),
                "items macro_f1 (mean)": items["macro_f1"].mean(),
                "items mae (mean)": items["mae"].mean(),
                "score rmse": regression["rmse"],
                "score pearson_r": regression["pearson_r"],
            }

    summary = pd.DataFrame(summary).T.rename_axis(["condition", "split"])
    per_task = pd.DataFrame(per_task)
    summary.to_csv(directory / "summary.csv")
    per_task.to_csv(directory / "per_task.csv", index=False)
    training.to_csv(directory / "training.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    conditions = list(predictions)
    print(f"Run: {directory.name}\n")
    print(
        "Item scores are per (comment, annotator) row; the score is per comment. baseline predicts train's most common\n"
        "rating for every item and train's mean score.\n"
    )
    print(summary.xs("test", level="split").loc[conditions].round(4).to_string())
    test_scores = per_task[per_task["split"] == "test"]
    for metric in ["macro_f1", "accuracy"]:
        print(f"\nPer item on test: {metric}")
        table = test_scores[test_scores["task"] != SCORE].pivot(index="task", columns="condition", values=metric)
        print(table.loc[ITEMS, conditions].round(3).to_string())
    print(f"\n{SCORE} on test, per comment")
    regression = test_scores[test_scores["task"] == SCORE].set_index("condition")[["rmse", "mae", "pearson_r"]]
    print(regression.loc[conditions].round(3).to_string())
    print("\nTraining\n")
    print(training.round(3).to_string())
    print(f"\nWrote summary.csv (val and test), per_task.csv and training.csv to {directory}")


def list_tasks(args):
    for i, run in enumerate(all_runs()):
        task_structure, tasks = model_spec(run)
        detail = task_type(run) if run in TASKS else f"task_structure {task_structure}, {len(tasks)} heads"
        print(f"{i:>2}  {condition_of(run):<13} {run:<18} {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", default=DEFAULTS["encoder"], help="Hugging Face model id")
    common.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Training seed (the splits are fixed)")
    common.add_argument("--run-name", help="Results subdirectory. Default: <encoder>_seed<seed>")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tasks", parents=[common], help="List run indices for the SLURM array").set_defaults(func=list_tasks)

    train_parser = commands.add_parser("train", parents=[common], help="Train one model")
    train_parser.add_argument("--task", help="Run name or index (see `tasks`). Default: $SLURM_ARRAY_TASK_ID")
    train_parser.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    train_parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    train_parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    train_parser.add_argument("--overwrite", action="store_true", help="Retrain even if this run's results exist")
    train_parser.set_defaults(func=train)

    commands.add_parser("aggregate", parents=[common], help="Score finished runs on val and test").set_defaults(
        func=aggregate
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
