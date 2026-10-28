"""
Train, validate and test a multitask model on the splits written by 01_clean.py.
One row per (comment, annotator): predicts that annotator's 10 item ratings plus the comment's hate_speech_score.
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

from multitask import MultiTaskModel, Trainer, TrainingConfig
from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig

CLEAN_DIR = Path(__file__).parent / "clean"

# Tiny BERT for CPU run; swap in roberta-large (Kennedy et al. 2020) for larger run.
ENCODER = "google/bert_uncased_L-2_H-128_A-2"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3

# Items are 0-4 except hatespeech (0-2)
# Kennedy et al. 2020 lists these as from mildest to most severe: Supportive, Counterspeech, Neutral, Bias, Hostility, Dehumanization, Violence, Genocide (Fig 1). Also high correlations between violence-genocide, and insult-respect (Fig 5)
# Additional task structure from manual correlation.
TASKS = ["sentiment", "respect", "insult", "humiliate", "status", "attack_defend", "dehumanize",  # hostility 
         "violence", "genocide",                                                                   # violence 
         "hatespeech", "hate_speech_score"]                                                        # overall-judgment trunk
task_structure=[[3], [7, 2, 2]]
task_types=["multiclass"] * 10 + ["regression"]
num_classes_per_task=[5] * 9 + [3, 1]


tokenizer = AutoTokenizer.from_pretrained(ENCODER)


def load(split, shuffle=False):
    df = pd.read_csv(CLEAN_DIR / f"hatespeech_{split}.csv")
    X = dict(
        tokenizer(df["text"].tolist(), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np")
    )
    y = {t: df[t].to_numpy("int32") for t in TASKS[:-1]}
    y["hate_speech_score"] = df["hate_speech_score"].to_numpy("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(df), seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":

    config = ModelConfig(
        task_structure=task_structure, # One flat branch with all 11 heads side by side (no hierarchy)
        task_names=TASKS, # Head names
        # 10 ordinal items treated as categorical (no worse than ordinal in Kennedy et al. 2020), plus the IRT score
        task_types=task_types,
        # 9 five-level items, hatespeech has 3 levels, regression has 1 output
        num_classes_per_task=num_classes_per_task,
        encoder_config=EncoderConfig(
            encoder_identifier=ENCODER,
            encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            # The tiny BERT only ships PyTorch weights, so convert them to TF on load
            from_pt=True,
        ),

        # Some hate_speech_score values are exactly -1.0, the default mask_value
        mask_value=-999.0,
    )
    trainer = Trainer(MultiTaskModel(config), TrainingConfig(epochs=EPOCHS, batch_size=BATCH_SIZE))
    trainer.fit(load("train", shuffle=True), load("val"), num_tasks=len(TASKS))

    results = trainer.evaluate(load("test"))
    print(pd.DataFrame(results["task_metrics"]).T)
