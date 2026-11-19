"""
NOTE: NEED TO CREATE SUBTOPICS NESTED UNDER TOPICS

Compare task structures for coding CAP party manifesto and platform sentences (UK, Ireland, US) by policy topic. Each
sentence has one major topic (22 classes, including 0 for no policy content) and one subtopic (226 classes) that sits
inside its major topic. Every model has the same encoder and shared layers, but the task_structure above them differs:

  single  Two separate models, one multiclass head each: one for the major topic and one for the subtopic. Nothing
          is shared, so this is the baseline with no multitask learning.
  joint   One model with both heads, each on its own branch off the shared layers: [[2]].

Besides each head's own scores, aggregate reports two things the hierarchy makes possible:
  major_from_sub  The major topic implied by the subtopic head: subtopic probabilities summed within each major topic.
  consistency     How often the predicted subtopic belongs to the predicted major topic.

Data: ../02_split.py splits by document (no manifesto spans two splits). Early stopping uses val, and test is the
evaluation split.

The script has two steps:

  train      Train one model and save its probabilities on val and test. --task is a run name or its index (`tasks`
             lists them: 0 majortopic, 1 subtopic, 2 joint), so a SLURM array can run all 3 jobs. Without --task it
             reads SLURM_ARRAY_TASK_ID.
  aggregate  Score every condition whose runs have all finished, on val and test. Takes seconds on CPU.

Examples:
  python compare_task_structures.py tasks
  python compare_task_structures.py train --task subtopic
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

# Task name -> the contiguous class index column written by ../01_clean.py
LABEL_COLUMNS = {"majortopic": "majortopic_idx", "subtopic": "subtopic_idx"}
TASKS = list(LABEL_COLUMNS)

JOINT_CONDITIONS = ["joint"]
CONDITIONS = ["single", *JOINT_CONDITIONS]

DEFAULTS = {
    "encoder": "roberta-base",
    "max_len": 64,  # 95% of sentences have at most 38 words
    "batch_size": 32,
    "epochs": 10,  # The upper limit: early stopping on validation loss (patience 3) usually stops sooner
    "learning_rate": 2e-5,  # The library default of 3e-4 is too high to fine-tune a full transformer
    "seed": 42,
}


# Data


def load(split):
    # keep_default_na=False so sentences like "None" or "NA" stay text
    df = pd.read_csv(CLEAN_DIR / f"cap_{split}.csv", keep_default_na=False)
    return df.astype({"majortopic_idx": int, "subtopic_idx": int})


def load_splits():
    return load("train"), load("val"), load("test")


def hierarchy(splits):
    """Return (classes per task, array mapping each subtopic index to its major topic index). The indices were made
    on the full dataset, so they're read from all splits together."""
    df = pd.concat(splits)
    num_classes = {task: int(df[column].max()) + 1 for task, column in LABEL_COLUMNS.items()}
    parents = df.groupby("subtopic_idx")["majortopic_idx"].agg(["first", "nunique"])
    assert (parents["nunique"] == 1).all(), "A subtopic appears under more than one major topic"
    parent_of_sub = np.full(num_classes["subtopic"], -1)
    parent_of_sub[parents.index] = parents["first"]
    return num_classes, parent_of_sub


def row_keys(df):
    # Document plus position in it. A string array, not pandas' object array, so np.load can read it back without pickle.
    return (df["doc_id"] + "_" + df.groupby("doc_id").cumcount().astype(str)).to_numpy(dtype=str)


# Task structures


def model_spec(run):
    """Return (task_structure, tasks) for one run, with tasks in the model's output order."""
    if run in TASKS:
        return [[1]], [run]
    if run == "joint":
        return [[len(TASKS)]], TASKS
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
    num_classes, _ = hierarchy([fit, val, test])
    run = resolve_run(args.task)
    out_dir = run_dir(args)
    out_path = out_dir / f"{run}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to retrain")
        return

    task_structure, tasks = model_spec(run)
    print(f"Run: {run} (condition {condition_of(run)}), task_structure {task_structure}, tasks: {', '.join(tasks)}")
    print(
        f"Sentences: {len(fit)} train, {len(val)} validation, {len(test)} test "
        f"({fit['doc_id'].nunique()}/{val['doc_id'].nunique()}/{test['doc_id'].nunique()} documents)"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def make_dataset(df, shuffle=False):
        X = dict(
            tokenizer(
                df["text"].tolist(), padding="max_length", truncation=True, max_length=args.max_len, return_tensors="np"
            )
        )
        y = {t: df[LABEL_COLUMNS[t]].to_numpy("int32") for t in tasks}
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(len(df), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    tf.keras.utils.set_random_seed(args.seed)
    config = ModelConfig(
        task_structure=task_structure,
        task_names=tasks,
        task_types=["multiclass"] * len(tasks),
        num_classes_per_task=[num_classes[t] for t in tasks],
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

    def probabilities(ds, split):
        outputs = trainer.model.predict(ds, verbose=0)
        # float16 halves the file: the subtopic head is 226 probabilities per sentence
        return {f"{split}_{t}": np.asarray(outputs[t]).astype("float16") for t in tasks}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        run=run,
        task_structure=str(task_structure),
        tasks=np.array(tasks),
        **probabilities(val_ds, "val"),
        **probabilities(test_ds, "test"),
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


def load_condition_probs(directory, splits):
    """Return {condition: {split: {task: probabilities}}} for every condition whose runs have all finished, plus a
    table of training statistics per run."""
    runs_by_condition = {"single": TASKS, **{c: [c] for c in JOINT_CONDITIONS}}
    probs, training, missing = {}, {}, []
    for condition, runs in runs_by_condition.items():
        absent = [r for r in runs if not (directory / f"{r}.npz").exists()]
        if absent:
            missing += absent
            continue
        merged = {split: {} for split in splits}
        for name in runs:
            run = np.load(directory / f"{name}.npz")
            for split, df in splits.items():
                # Guard against mixing runs from different data or splits
                assert (run[f"{split}_keys"] == row_keys(df)).all(), f"{name}: {split} sentences differ from this split"
                for t in run["tasks"]:
                    merged[split][str(t)] = run[f"{split}_{t}"].astype("float32")
            training[name] = {
                "condition": condition,
                "epochs": int(run["epochs_trained"]),
                "best_val_loss": float(run["best_val_loss"]),
                "minutes": float(run["train_seconds"]) / 60,
            }
        assert sorted(merged["test"]) == sorted(TASKS), f"{condition}: runs don't cover the tasks exactly once"
        probs[condition] = merged

    if missing:
        print(f"Not scored yet, {len(missing)} runs missing in {directory}: {', '.join(missing)}\n")
    if not probs:
        raise SystemExit("No condition has all its runs")
    return probs, pd.DataFrame(training).T


def baseline_probs(fit, n, num_classes):
    """Predict train's most common class for every sentence, the floor any model should beat."""
    out = {}
    for task, column in LABEL_COLUMNS.items():
        p = np.zeros((n, num_classes[task]), dtype="float32")
        p[:, fit[column].mode()[0]] = 1
        out[task] = p
    return out


def classification_scores(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        # Averaged over the classes that occur in this split or are predicted in it
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }


def aggregate(args):
    fit, val, test = load_splits()
    splits = {"val": val, "test": test}
    num_classes, parent_of_sub = hierarchy([fit, val, test])
    directory = run_dir(args)
    probs, training = load_condition_probs(directory, splits)
    probs = {"baseline": {s: baseline_probs(fit, len(df), num_classes) for s, df in splits.items()}, **probs}

    # Sums subtopic probabilities into their major topics: [subtopics, major topics]
    sub_to_major = np.zeros((num_classes["subtopic"], num_classes["majortopic"]), dtype="float32")
    sub_to_major[np.arange(num_classes["subtopic"]), parent_of_sub] = 1

    scores = []
    for condition, by_split in probs.items():
        for split, df in splits.items():
            major_true, sub_true = df["majortopic_idx"].to_numpy(), df["subtopic_idx"].to_numpy()
            major_pred = by_split[split]["majortopic"].argmax(axis=1)
            sub_pred = by_split[split]["subtopic"].argmax(axis=1)
            major_from_sub = (by_split[split]["subtopic"] @ sub_to_major).argmax(axis=1)
            for target, y, pred in [
                ("majortopic", major_true, major_pred),
                ("subtopic", sub_true, sub_pred),
                ("major_from_sub", major_true, major_from_sub),
            ]:
                scores.append({"condition": condition, "split": split, "target": target, **classification_scores(y, pred)})
            scores.append(
                {
                    "condition": condition,
                    "split": split,
                    "target": "consistency",
                    "accuracy": float((parent_of_sub[sub_pred] == major_pred).mean()),
                }
            )

    scores = pd.DataFrame(scores)
    scores.to_csv(directory / "scores.csv", index=False)
    training.to_csv(directory / "training.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    conditions = list(probs)
    print(f"Run: {directory.name}\n")
    print(
        "baseline predicts train's most common class. major_from_sub is the major topic implied by the subtopic head;\n"
        "consistency is the share of sentences whose predicted subtopic belongs to the predicted major topic.\n"
    )
    columns = [(t, m) for t in ["majortopic", "subtopic", "major_from_sub"] for m in ["accuracy", "macro_f1"]]
    for split in ["test", "val"]:
        table = scores[scores["split"] == split].pivot(index="condition", columns="target", values=["accuracy", "macro_f1"])
        table = table.swaplevel(axis=1)[columns + [("consistency", "accuracy")]].loc[conditions]
        table.columns = [f"{t} {m}" if t != "consistency" else "consistency" for t, m in table.columns]
        print(f"On {split} (weighted F1 is in scores.csv)\n{table.round(4).to_string()}\n")
    print("Training\n")
    print(training.round(3).to_string())
    print(f"\nWrote scores.csv (val and test) and training.csv to {directory}")


def list_tasks(args):
    fit, val, test = load_splits()
    num_classes, _ = hierarchy([fit, val, test])
    for i, run in enumerate(all_runs()):
        task_structure, tasks = model_spec(run)
        detail = ", ".join(f"{t} ({num_classes[t]} classes)" for t in tasks)
        print(f"{i:>2}  {condition_of(run):<7} {run:<11} task_structure {task_structure}: {detail}")


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
