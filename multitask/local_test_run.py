"""
Local test run script for multitask model.

Run this script from the project root directory:
    python -m multitask.local_test_run
    OR
    cd /Users/jgint/Documents/code/multitask && python multitask/local_test_run.py
"""

from multitask import ModelConfig, TrainingConfig, MultiTaskModel, Trainer
from multitask import load_model

import pandas as pd
import tensorflow as tf
import numpy as np
import os

def dataframe_to_dataset(df, task_names, batch_size=32, shuffle=False):
    """Convert pandas DataFrame to tf.data.Dataset for Universal Sentence Encoder.
    
    Args:
        df: DataFrame with 'text' column and task label columns
        task_names: List of task names matching the columns in df
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        
    Returns:
        tf.data.Dataset yielding (text_input, labels_dict) tuples
    """
    # Extract texts and labels
    # For Universal Sentence Encoder, we need to ensure texts are strings
    texts = df['text'].astype(str).tolist()  # Convert to list to preserve strings
    labels_dict = {}
    
    for task_name in task_names:
        if task_name in df.columns:
            labels = df[task_name].values
            # Handle missing values (replace NaN with -1 for masking)
            labels = np.where(np.isnan(labels), -1, labels)
            labels_dict[task_name] = labels.astype(np.float32)
        else:
            raise ValueError(f"Task column '{task_name}' not found in DataFrame")
    
    # Create dataset from tensors
    # Convert texts to TensorFlow string tensor explicitly
    texts_tensor = tf.constant(texts, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((texts_tensor, labels_dict))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Configure model with mixed tasks
# NOTE: Update task names and types to match actual CSV columns
# The CSV has columns like: disagree_agree, attacking_respectful, emotion_fact, etc.
# Using just the three for demonstration
tasks = {
    'disagree_agree_bi': {'type': 'binary', 'num_classes': 2},  # Binary: 2 classes, but outputs 1 dimension
    'attacking_respectful_bi': {'type': 'binary', 'num_classes': 2},
    'emotion_fact_bi': {'type': 'binary', 'num_classes': 2},
}

# ModelConfig requires task_structure to match the number of tasks
# task_structure=[[3]] means: 3 branches at level 0, no further branching = 3 total outputs
config = ModelConfig(
    task_structure=[[3]],  # 3 outputs (one per task)
    task_names=list(tasks.keys()),
    task_types=[task_desc['type'] for task_desc in tasks.values()],
    num_classes_per_task=[task_desc['num_classes'] for task_desc in tasks.values()],
)

# Step 1: Load and prepare data
# Determine data path so code is runnable from root or multitask directory

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'multitask' else script_dir
data_path = os.path.join(project_root, 'multitask', 'data')

print("Loading data...")
data = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_train.csv')) # training data
data_val = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_val.csv')) # evaluation data

# Get task names from config
task_names = config.task_names  # This is a list, not a dict
num_tasks = len(task_names)

# Convert DataFrames to tf.data.Dataset
print("Converting DataFrames to datasets...")
train_dataset = dataframe_to_dataset(data, task_names, batch_size=32, shuffle=True)
val_dataset = dataframe_to_dataset(data_val, task_names, batch_size=32, shuffle=False)

# Step 2: Check if model already exists
# Avoids having to recreate model unnecessarily, e.g. during testing
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'my_model')
weights_path = os.path.join(model_path, 'weights.weights.h5')
config_path = os.path.join(model_path, 'config.json')
metadata_path = os.path.join(model_path, 'metadata.json')
model_exists = all(os.path.exists(p) for p in [weights_path, config_path, metadata_path])

if model_exists:
    print(f"Found existing model at {model_path}, skipping training...")
    print("Loading model for predictions...")
    model, thresholds = load_model(model_path)
else:
    # Step 2: Create and train the model
    print("Creating model...")
    # Pass encoder to MultiTaskModel
    model = MultiTaskModel(
        config,
        encoder='https://tfhub.dev/google/universal-sentence-encoder/4',
        trainable_encoder=False  # Freeze encoder weights
    )

    training_config = TrainingConfig()
    trainer = Trainer(model, training_config)

    print("Training model...")
    history, thresholds = trainer.fit(
        train_dataset, 
        val_dataset, 
        num_tasks=num_tasks,
        optimize_thresholds=True
    )

    # Step 3: Save the trained model with thresholds
    print("Saving model...")
    model.save(model_path, thresholds=thresholds)

    # Step 4: Load the saved model for predictions
    print("Loading model for predictions...")
    model, thresholds = load_model(model_path)

# Step 5: Use the loaded model for predictions
print("Making predictions...")
data_test = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_test.csv'))

# For prediction, we only need the text inputs (no labels)
# Convert to list first, then to TensorFlow string tensor
texts_list = data_test['text'].astype(str).tolist()
texts_tensor = tf.constant(texts_list, dtype=tf.string)
texts_only = tf.data.Dataset.from_tensor_slices(texts_tensor)
texts_only = texts_only.batch(32).prefetch(tf.data.AUTOTUNE)

predictions = model.predict(texts_only)  # Dict of arrays per task
print(f"Predictions completed!")
print(f"Prediction shapes: {[f'{k}: {v.shape}' for k, v in predictions.items()]}")