# multitask

A flexible multi-task learning library for NLP supporting mixed task types (binary/multiclass/multilabel), multiple encoders, and distributed training.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Encoder Types](#encoder-types)
- [Training](#training)
- [Model Persistence](#model-persistence)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation and License](#citation)

## Features

- Mixed task types in single model (binary, multiclass, multilabel)
- Multiple encoders: TensorFlow Hub, HuggingFace, pre-computed embeddings
- Automatic class imbalance and task weighting
- Automatic threshold optimization for binary/multilabel tasks
- Cross-platform distributed training with automatic GPU detection
- Model persistence with threshold saving
- Comprehensive type hints

## Installation

First, create a folder `multitask/data` and populate with the data provided. Ask someone for a dropbox link.

**Local development installation:**
```bash
# Basic (for development only - change before distributing)
pip install -e .

# With optional encoder support
pip install tensorflow-hub transformers
```

**Conda installation:**
```
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate multitask
```

**Platform-specific GPU support:**
- Apple Silicon: `pip install tensorflow-macos tensorflow-metal`
- NVIDIA: `pip install tensorflow[and-cuda]`



## Quick Start

```python
from multitask import ModelConfig, TrainingConfig, MultiTaskModel, Trainer
import pandas as pd

# Configure model with mixed tasks
config = ModelConfig(
    tasks={
        'sentiment': {'type': 'multiclass', 'num_classes': 3},
        'toxicity': {'type': 'binary', 'num_classes': 1},
        'emotions': {'type': 'multilabel', 'num_classes': 5},
    },
    encoder_type='tfhub',
    encoder_name='https://tfhub.dev/google/universal-sentence-encoder/4',
)

# Train
data = pd.read_csv('data.csv')  # Needs: text, sentiment, toxicity, emotions columns
trainer = Trainer(MultiTaskModel(config), TrainingConfig())
history, thresholds = trainer.fit(data, data_val)

# Predict
predictions = model.predict(data_test)  # Dict of arrays per task
```

## Data Format

### Text Input

DataFrame with `text` column and one column per task:

```python
import pandas as pd

data = pd.DataFrame({
    'text': ['great product', 'terrible', ...],
    'sentiment': [2, 0, ...],              # 0-2 for 3-class
    'toxicity': [0, 1, ...],               # 0-1 for binary
    'emotions': ['joy anger', 'fear', ...],  # Space-separated for multilabel
})
```

### Pre-computed Embeddings

Add `embedding` column with numpy arrays:

```python
import numpy as np

data['embedding'] = [np.random.randn(768) for _ in range(len(data))]

config = ModelConfig(
    tasks={'sentiment': {'type': 'multiclass', 'num_classes': 3}},
    encoder_type='from_embedding',
    embedding_dim=768,
)
```

### Missing Labels

Use `-1` to mark missing labels (excluded from loss):

```python
data = pd.DataFrame({
    'text': ['text1', 'text2', 'text3'],
    'task1': [0, 1, -1],     # task1 missing for text3
    'task2': [1, -1, 0],     # task2 missing for text2
})
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for complete documentation.

## Encoder Types

### TensorFlow Hub

```python
config = ModelConfig(
    tasks={'sentiment': {'type': 'multiclass', 'num_classes': 3}},
    encoder_type='tfhub',
    encoder_name='https://tfhub.dev/google/universal-sentence-encoder/4',
)
```

### HuggingFace

```python
config = ModelConfig(
    tasks={'sentiment': {'type': 'multiclass', 'num_classes': 3}},
    encoder_type='huggingface',
    encoder_name='bert-base-uncased',
    trainable_encoder=False,  # Freeze weights
)
```

### Pre-computed Embeddings

```python
config = ModelConfig(
    tasks={'sentiment': {'type': 'multiclass', 'num_classes': 3}},
    encoder_type='from_embedding',
    encoder_name='pre_computed',
    embedding_dim=768,
)
```

See [ENCODER_SUPPORT.md](docs/ENCODER_SUPPORT.md) for all options.

## Training

### Basic

```python
config = TrainingConfig(batch_size=32, epochs=10, learning_rate=2e-5)
trainer = Trainer(model, config)
history, thresholds = trainer.fit(train_data, val_data)
```

### Automatic Weight Computation

By default, trainer automatically computes:
- **Task weights**: Inversely proportional to non-masked samples per task
- **Class weights**: Inversely proportional to class frequency

Override with explicit weights:

```python
trainer.fit(
    train_data, val_data,
    class_weights=[{0: 1.0, 1: 2.0}, ...],  # Per task
    task_weights=[1.0, 2.0, ...],
)
```

### Threshold Optimization

Find optimal thresholds for binary/multilabel tasks:

```python
history, thresholds = trainer.fit(
    train_data, val_data,
    optimize_thresholds=True,  # Uses Youden's J statistic
)
```

### Distributed Training

Automatic GPU detection and strategy selection:

```python
from multitask import get_distribution_strategy, setup_gpu_memory_growth

setup_gpu_memory_growth()
strategy, info = get_distribution_strategy('auto', verbose=True)

with strategy.scope():
    model = MultiTaskModel(config)
    trainer = Trainer(model, training_config, strategy=strategy)

trainer.fit(train_data, val_data)
```

See [DISTRIBUTED_TRAINING.md](docs/DISTRIBUTED_TRAINING.md).

### Checkpointing

```python
trainer.fit(
    train_data, val_data,
    checkpoint_dir='checkpoints/exp1',
    checkpoint_monitor='val_loss',
)
```

### Verbosity

```python
trainer.fit(train_data, val_data, verbose=0)  # Silent
trainer.fit(train_data, val_data, verbose=1)  # Progress (default)
trainer.fit(train_data, val_data, verbose=2)  # One line per epoch
```

## Model Persistence

### Save

```python
model.save('models/my_model', thresholds=thresholds)
```

### Load

```python
from multitask import load_model

model, thresholds = load_model('models/my_model')
predictions = model.predict(test_data)
```

See [MODEL_PERSISTENCE.md](docs/MODEL_PERSISTENCE.md).

## Configuration

### ModelConfig

```python
ModelConfig(
    tasks={...},                      # Task definitions
    encoder_type='tfhub',             # 'tfhub', 'huggingface', 'from_embedding'
    encoder_name='...',               # URL or model name
    embedding_dim=768,                # For from_embedding only
    shared_hidden_layers=[256, 128],  # Shared layers
    task_hidden_layers=[64],          # Task-specific layers
    dropout=0.1,
    trainable_encoder=False,
)
```

### TrainingConfig

```python
TrainingConfig(
    batch_size=32,
    epochs=10,
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    dropout=0.1,
    early_stopping=True,
    early_stopping_patience=3,
)
```

## Troubleshooting

**Import errors:**
```bash
pip install tensorflow-hub         # TF Hub
pip install transformers           # HuggingFace
```

**GPU issues:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Out of memory:**
```python
from multitask import setup_gpu_memory_growth
setup_gpu_memory_growth()
# Or: TrainingConfig(batch_size=16)
```

**NaN loss:**
- Check labels are integers in valid range
- Reduce learning rate: `learning_rate=1e-5`

## API Reference

**Classes:** `MultiTaskModel`, `Trainer`, `ModelConfig`, `TrainingConfig`

**Utils:** `load_model()`, `get_distribution_strategy()`, `setup_gpu_memory_growth()`, `check_distributed_compatibility()`, `print_device_info()`

## Examples

- `examples/save_load_example.py`: Model persistence
- `examples/distributed_training.py`: Distributed training

## Citation

```bibtex
@software{multitask,
  title = {multitask: A Flexible Multi-Task Learning Library for NLP},
  author = {Mehlhaff, Isaac D. and Morucci, Marco},
  year = {2025},
}
```

See [LICENSE](LICENSE).

Pull requests welcome.
