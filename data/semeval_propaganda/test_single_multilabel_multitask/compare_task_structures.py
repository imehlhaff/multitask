"""
Compare five task structures for predicting the 19 persuasion techniques that occur in English SemEval 2023 Task 3
subtask 3. Every model has the same encoder and shared layers, but the task_structure above them differs:

  single           19 separate models, one binary head each. Nothing is shared between techniques, so this is the
                   baseline with no multitask learning.
  multilabel       One model with one multilabel head that outputs all 19 techniques as a vector of 19 values.
  category_heads   One model with six multilabel heads, one per task category, each over that category's techniques.
  category_trunks  One model with a shared branch per task category, each splitting into one binary head per
                   technique in that category.
  core_trunk       One model where the four techniques that co-occur most (see label_correlations.ipynb) share a
                   trunk before splitting into their own binary heads. The other 15 techniques each get their own
                   branch and binary head.

All five use the same hyperparameters, validation split and thresholding rule, and are scored on dev with the
official SemEval scorer. Test has no gold labels, so dev is the evaluation split. Early stopping and thresholds use a
validation split carved out of train by article, so no dev label is seen before scoring.

The script has two steps:

  train      Train one model and save its probabilities on the validation split and on dev. --task is a run name
             or its index (`tasks` lists them: 0-18 single techniques, then multilabel, category_heads,
             category_trunks, core_trunk), so a SLURM array can run all 23 jobs. Without --task it reads
             SLURM_ARRAY_TASK_ID.
  aggregate  Pick each technique's threshold on the validation split, predict dev, and score every condition whose
             runs have all finished. Takes seconds on CPU.

Examples:
  python compare_task_structures.py tasks
  python compare_task_structures.py train --task Loaded_Language
  python compare_task_structures.py train --task core_trunk
  python compare_task_structures.py aggregate
"""
import argparse
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent
CLEAN_DIR = DATA_DIR / "clean"
BUNDLE_DIR = DATA_DIR / "semeval2023task3bundle-v4"
SCORER_PATH = BUNDLE_DIR / "scorers" / "scorer-subtask-3.py"
OFFICIAL_TECHNIQUES_PATH = BUNDLE_DIR / "scorers" / "techniques_subtask3.txt"
DEV_GOLD_PATH = BUNDLE_DIR / "data" / "en" / "dev-labels-subtask-3.txt"
RESULTS_DIR = HERE / "results"

NON_TECHNIQUE_COLUMNS = ["text", "technique", "article_id", "paragraph_id"]

# The six technique categories of the task (Piskorski et al. 2023), for all 23 official techniques
CATEGORY = {
    "Appeal_to_Authority": "Justification",
    "Appeal_to_Popularity": "Justification",
    "Appeal_to_Values": "Justification",
    "Appeal_to_Fear-Prejudice": "Justification",
    "Flag_Waving": "Justification",
    "Causal_Oversimplification": "Simplification",
    "False_Dilemma-No_Choice": "Simplification",
    "Consequential_Oversimplification": "Simplification",
    "Straw_Man": "Distraction",
    "Red_Herring": "Distraction",
    "Whataboutism": "Distraction",
    "Slogans": "Call",
    "Appeal_to_Time": "Call",
    "Conversation_Killer": "Call",
    "Loaded_Language": "Manipulative wording",
    "Repetition": "Manipulative wording",
    "Exaggeration-Minimisation": "Manipulative wording",
    "Obfuscation-Vagueness-Confusion": "Manipulative wording",
    "Name_Calling-Labeling": "Attack on reputation",
    "Doubt": "Attack on reputation",
    "Guilt_by_Association": "Attack on reputation",
    "Appeal_to_Hypocrisy": "Attack on reputation",
    "Questioning_the_Reputation": "Attack on reputation",
}

CORE = ["Loaded_Language", "Name_Calling-Labeling", "Exaggeration-Minimisation", "Doubt"]

# Conditions trained as one model each; "single" is the 19 separate models
JOINT_CONDITIONS = ["multilabel", "category_heads", "category_trunks", "core_trunk"]
CONDITIONS = ["single", *JOINT_CONDITIONS]

# Held out from train for early stopping and thresholds. The split is deterministic, so all 23 jobs and every training
# seed use the same paragraphs. Several techniques have under 20 train paragraphs, so a random 10% of articles can
# miss them entirely; the split is the first one where every technique has at least MIN_VAL_POSITIVES.
VAL_FRACTION = 0.1
MIN_VAL_POSITIVES = 2

DEFAULTS = {
    "encoder": "roberta-base",
    "max_len": 256,  # 95% of train paragraphs have at most 112 words
    "batch_size": 32,
    "epochs": 10,  # The upper limit: early stopping on validation loss (patience 3) usually stops sooner
    "learning_rate": 2e-5,  # The library default of 3e-4 is too high to fine-tune a full transformer
    "seed": 42,
}


# Data


def load(split):
    return pd.read_csv(CLEAN_DIR / f"semeval_{split}.csv", dtype={"article_id": str})


def techniques_in(train):
    """The technique columns that vary in train, in the official order. In English 4 of the 23 never occur."""
    columns = [c for c in train.columns if c not in NON_TECHNIQUE_COLUMNS]
    return [c for c in columns if train[c].nunique() > 1]


def split_by_article(train, techniques):
    """Hold out whole articles, since paragraphs from the same article share topic and wording."""
    for split_seed in range(1000):
        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=split_seed)
        fit_index, val_index = next(splitter.split(train, groups=train["article_id"]))
        if train.iloc[val_index][techniques].sum().min() >= MIN_VAL_POSITIVES:
            return train.iloc[fit_index].reset_index(drop=True), train.iloc[val_index].reset_index(drop=True)
    raise RuntimeError(f"No article split gives every technique {MIN_VAL_POSITIVES} validation positives")


def paragraph_keys(df):
    # A string array, not pandas' object array, so np.load can read it back without pickle
    return (df["article_id"] + "_" + df["paragraph_id"].astype(str)).to_numpy(dtype=str)


def load_splits():
    train, dev = load("train"), load("dev")
    techniques = techniques_in(train)
    fit, val = split_by_article(train, techniques)
    return techniques, fit, val, dev


# Task structures


@dataclass
class Head:
    name: str
    type: str  # "binary" or "multilabel"
    columns: list  # The technique columns this head predicts, in output order


def model_spec(run, techniques):
    """Return (task_structure, heads) for one run, with heads in the model's output order."""
    if run in techniques:
        return [[1]], [Head(run, "binary", [run])]

    if run == "multilabel":
        return [[1]], [Head("multilabel", "multilabel", techniques)]

    # The six categories in the order their first technique appears, each with its techniques
    members = {c: [t for t in techniques if CATEGORY[t] == c] for c in dict.fromkeys(CATEGORY[t] for t in techniques)}

    if run == "category_heads":
        # Six branches off the shared layers, each ending in one multilabel head over its category's techniques
        heads = [Head("category_" + c.lower().replace(" ", "_"), "multilabel", ts) for c, ts in members.items()]
        return [[len(heads)]], heads

    if run == "category_trunks":
        # Level 0: one branch per category. Level 1: each category branch splits into one branch per technique.
        # Each final branch ends in a binary head. In English this is [[6], [4, 2, 3, 2, 4, 4]].
        heads = [Head(t, "binary", [t]) for ts in members.values() for t in ts]
        return [[len(members)], [len(ts) for ts in members.values()]], heads

    if run == "core_trunk":
        # Level 0: one branch for the core group plus one per other technique. Level 1: the core branch splits into
        # four, every other branch continues as one. Each final branch ends in a binary head.
        rest = [t for t in techniques if t not in CORE]
        heads = [Head(t, "binary", [t]) for t in CORE + rest]
        return [[1 + len(rest)], [len(CORE)] + [1] * len(rest)], heads

    raise ValueError(run)


def all_runs(techniques):
    return [*techniques, *JOINT_CONDITIONS]


def condition_of(run):
    return run if run in JOINT_CONDITIONS else "single"


def resolve_run(task, techniques):
    runs = all_runs(techniques)
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

    techniques, fit, val, dev = load_splits()
    run = resolve_run(args.task, techniques)
    out_dir = run_dir(args)
    out_path = out_dir / f"{run}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to retrain")
        return

    task_structure, heads = model_spec(run, techniques)
    print(f"Run: {run} (condition {condition_of(run)}), task_structure {task_structure}")
    for head in heads:
        print(f"  {head.type:<10} head {head.name}: {', '.join(head.columns)}")
    print(f"Paragraphs: {len(fit)} train, {len(val)} validation ({val['article_id'].nunique()} articles), {len(dev)} dev")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def labels(df):
        return {
            head.name: df[head.columns].to_numpy("float32") if head.type == "multilabel" else df[head.columns[0]].to_numpy("float32")
            for head in heads
        }

    def make_dataset(df, shuffle=False):
        X = dict(
            tokenizer(
                df["text"].tolist(), padding="max_length", truncation=True, max_length=args.max_len, return_tensors="np"
            )
        )
        ds = tf.data.Dataset.from_tensor_slices((X, labels(df)))
        if shuffle:
            ds = ds.shuffle(len(df), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    tf.keras.utils.set_random_seed(args.seed)
    config = ModelConfig(
        task_structure=task_structure,
        task_names=[head.name for head in heads],
        task_types=[head.type for head in heads],
        num_classes_per_task=[2 if head.type == "binary" else len(head.columns) for head in heads],
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
    )
    fit_ds, val_ds, dev_ds = make_dataset(fit, shuffle=True), make_dataset(val), make_dataset(dev)

    trainer = Trainer(MultiTaskModel(config), training_config)
    start = time.time()
    history, _ = trainer.fit(fit_ds, val_ds, num_tasks=len(heads))
    seconds = time.time() - start

    def probabilities(ds, n):
        """One column per technique, in the order of `columns` below."""
        outputs = trainer.model.predict(ds, verbose=0)
        return np.hstack([np.asarray(outputs[head.name]).reshape(n, -1) for head in heads])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        run=run,
        task_structure=str(task_structure),
        columns=np.array([c for head in heads for c in head.columns]),
        val_probs=probabilities(val_ds, len(val)),
        dev_probs=probabilities(dev_ds, len(dev)),
        val_keys=paragraph_keys(val),
        dev_keys=paragraph_keys(dev),
        epochs_trained=len(history.history["loss"]),
        best_val_loss=min(history.history["val_loss"]),
        train_seconds=seconds,
        encoder=args.encoder,
        seed=args.seed,
    )
    print(f"Trained {len(history.history['loss'])} epochs in {seconds / 60:.1f} min. Wrote {out_path}")


# Aggregate


def load_scorer():
    """Import the official scorer as a module (it only runs its command-line code under __main__)."""
    spec = importlib.util.spec_from_file_location("semeval_scorer_subtask3", SCORER_PATH)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def best_f1_threshold(y_true, probs):
    """The threshold that maximises F1 on the validation split. Falls back to 0.5 when there are no positives."""
    if y_true.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # The last precision/recall pair has no threshold
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def write_scorer_file(dev, predictions, techniques, path):
    """One line per paragraph: article_id<TAB>paragraph_id<TAB>technique1,technique2 (empty if none)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for article_id, paragraph_id, row in zip(dev["article_id"], dev["paragraph_id"], predictions):
            names = ",".join(t for t, predicted in zip(techniques, row) if predicted)
            f.write(f"{article_id}\t{paragraph_id}\t{names}\n")


def load_condition_probs(directory, techniques, val, dev):
    """Return {condition: (val_probs, dev_probs)}, each [paragraphs, techniques] in technique order, for every
    condition whose runs have all finished, plus a table of training statistics per run."""
    runs_by_condition = {"single": techniques, **{c: [c] for c in JOINT_CONDITIONS}}
    probs, training, missing = {}, {}, []
    for condition, runs in runs_by_condition.items():
        absent = [r for r in runs if not (directory / f"{r}.npz").exists()]
        if absent:
            missing += absent
            continue
        val_parts, dev_parts, columns = [], [], []
        for name in runs:
            run = np.load(directory / f"{name}.npz")
            # Guard against mixing runs from different data or splits
            assert (run["val_keys"] == paragraph_keys(val)).all(), f"{name}: validation paragraphs differ from this split"
            assert (run["dev_keys"] == paragraph_keys(dev)).all(), f"{name}: dev paragraphs differ"
            val_parts.append(run["val_probs"])
            dev_parts.append(run["dev_probs"])
            columns += [str(c) for c in run["columns"]]
            training[name] = {
                "condition": condition,
                "epochs": int(run["epochs_trained"]),
                "best_val_loss": float(run["best_val_loss"]),
                "minutes": float(run["train_seconds"]) / 60,
            }
        assert sorted(columns) == sorted(techniques), f"{condition}: runs don't cover the techniques exactly once"
        order = [columns.index(t) for t in techniques]
        probs[condition] = (np.hstack(val_parts)[:, order], np.hstack(dev_parts)[:, order])

    if missing:
        print(f"Not scored yet, {len(missing)} runs missing in {directory}: {', '.join(missing)}\n")
    if not probs:
        raise SystemExit("No condition has all its runs")
    return probs, pd.DataFrame(training).T


def aggregate(args):
    techniques, _, val, dev = load_splits()
    directory = run_dir(args)
    probs, training = load_condition_probs(directory, techniques, val, dev)
    scorer = load_scorer()
    official_techniques = scorer.read_techniques_list_from_file(OFFICIAL_TECHNIQUES_PATH)
    dev_gold_file = scorer._read_csv_input_file(DEV_GOLD_PATH)

    val_gold = val[techniques].to_numpy()
    dev_gold = dev[techniques].to_numpy()
    rare = {t: int(n) for t, n in zip(techniques, val_gold.sum(axis=0)) if n < 5}
    if rare:
        print(f"Warning: thresholds for these techniques rest on fewer than 5 validation positives: {rare}\n")

    summary, per_technique = {}, []
    for condition, (val_probs, dev_probs) in probs.items():
        tuned = np.array([best_f1_threshold(val_gold[:, j], val_probs[:, j]) for j in range(len(techniques))])
        for rule, thresholds in {"tuned": tuned, "0.5": np.full(len(techniques), 0.5)}.items():
            predictions = dev_probs >= thresholds
            path = directory / f"dev_predictions_{condition}_{rule}.txt"
            write_scorer_file(dev, predictions, techniques, path)

            # The official score, read back from the file the scorer would read
            pred_file = scorer._read_csv_input_file(path)
            assert scorer.correct_format(pred_file, dev_gold_file, official_techniques)
            official_macro, official_micro = scorer.evaluate(pred_file, dev_gold_file, official_techniques)
            summary[(condition, rule)] = {
                "micro_f1 (official)": official_micro,
                "macro_f1 (official, 23)": official_macro,
                f"macro_f1 ({len(techniques)})": f1_score(dev_gold, predictions, average="macro", zero_division=1),
                "labels predicted": int(predictions.sum()),
            }

        predictions = dev_probs >= tuned
        for j, t in enumerate(techniques):
            per_technique.append(
                {
                    "technique": t,
                    "category": CATEGORY[t],
                    "dev positives": int(dev_gold[:, j].sum()),
                    "condition": condition,
                    "threshold": tuned[j],
                    "precision": precision_score(dev_gold[:, j], predictions[:, j], zero_division=0),
                    "recall": recall_score(dev_gold[:, j], predictions[:, j], zero_division=0),
                    "f1": f1_score(dev_gold[:, j], predictions[:, j], zero_division=0),
                    # Threshold-free, so it separates how well the model ranks paragraphs from where the threshold
                    # landed
                    "avg_precision": (
                        average_precision_score(dev_gold[:, j], dev_probs[:, j]) if dev_gold[:, j].any() else np.nan
                    ),
                }
            )

    summary = pd.DataFrame(summary).T.rename_axis(["condition", "thresholds"])
    per_technique = pd.DataFrame(per_technique)
    summary.to_csv(directory / "summary.csv")
    per_technique.to_csv(directory / "per_technique.csv", index=False)
    training.to_csv(directory / "training.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"Run: {directory.name}\n")
    print(
        "Dev scores. The official macro-F1 averages over all 23 techniques with zero_division=1, so each of the 4\n"
        "techniques that never occur in English adds a free 1.0. The 19-technique macro-F1 leaves them out.\n"
    )
    print(summary.round(4).to_string())
    for metric in ["f1", "avg_precision"]:
        print(f"\nPer technique on dev: {metric}" + (" (thresholds tuned on validation)" if metric == "f1" else ""))
        table = per_technique.pivot_table(
            index=["technique", "category", "dev positives"], columns="condition", values=metric, sort=False
        )[list(probs)]
        print(table.sort_index(level="dev positives", ascending=False, sort_remaining=False).round(3).to_string())
    print("\nTraining\n")
    print(training.round(3).to_string())
    print(f"\nWrote summary.csv, per_technique.csv, training.csv and dev prediction files to {directory}")


def list_tasks(args):
    techniques, fit, val, _ = load_splits()
    for i, run in enumerate(all_runs(techniques)):
        task_structure, heads = model_spec(run, techniques)
        if run in techniques:
            detail = f"{fit[run].sum()} train / {val[run].sum()} validation positives"
        else:
            detail = f"task_structure {task_structure}, {len(heads)} heads"
        print(f"{i:>2}  {condition_of(run):<15} {run:<33} {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", default=DEFAULTS["encoder"], help="Hugging Face model id")
    common.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Training seed (the validation split is fixed)")
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

    commands.add_parser("aggregate", parents=[common], help="Score finished runs on dev").set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
