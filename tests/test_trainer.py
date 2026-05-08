import pytest
import tensorflow as tf
import numpy as np
from multitask.trainers.trainer import Trainer
from multitask.config import TrainingConfig, ModelConfig
from multitask.models.hierarchical import MultiTaskModel


@pytest.fixture
def dummy_trainer():
    """Initialize a trainer with a 3-task ModelConfig matching the research scope."""
    config = ModelConfig(
        task_structure=[[3]], 
        task_names=["task_binary", "task_multiclass", "task_multilabel"],
        task_types=["binary", "multiclass", "multilabel"],
        num_classes_per_task=[2, 3, 2]
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())
    return trainer


def test_loss_dictionary_direct_access(dummy_trainer):
    """Verifies that the loss correctly pulls from y_true_dict keys."""
    y_true = {
        "task_binary": tf.constant([[1.0]], dtype=tf.float32),
        "task_multiclass": tf.constant([2], dtype=tf.int32), 
        "task_multilabel": tf.constant([[1.0, 0.0]], dtype=tf.float32)
    }
    y_pred = {
        "task_binary": tf.constant([[0.8]], dtype=tf.float32),
        "task_multiclass": tf.constant([[0.1, 0.1, 0.8]], dtype=tf.float32),
        "task_multilabel": tf.constant([[0.7, 0.2]], dtype=tf.float32)
    }
    
    loss_fn = dummy_trainer._build_loss_function(
        num_tasks=3, 
        class_weights=[{0: 1.0, 1: 1.0, 2: 1.0}] * 3, 
        task_weights=[1.0, 1.0, 1.0]
    )
    
    loss = loss_fn(y_true, y_pred)
    assert loss.dtype == tf.float32
    assert loss > 0


def test_masking_and_normalization_logic():
    """Verifies that mask_value (-1) is excluded from loss and averaging."""
    # Create a trainer with only 1 task for this specific test
    config = ModelConfig(
        task_structure=[[1]], 
        task_names=["task_binary"],
        task_types=["binary"],
        num_classes_per_task=[2]
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())
    
    # 2 labels - one is masked
    y_true = {'task_binary': tf.constant([[1.0], [-1.0]], dtype=tf.float32)}
    y_pred = {'task_binary': tf.constant([[0.5], [0.5]], dtype=tf.float32)}
    
    loss_fn = trainer._build_loss_function(
        num_tasks=1, 
        class_weights=[{0: 1.0, 1: 1.0}], 
        task_weights=[1.0]
    )
    
    loss = loss_fn(y_true, y_pred)
    # (0.693 * 1.0 + 0.0) / 1.0 valid label = 0.693
    assert loss.numpy() == pytest.approx(0.693, abs=0.01)


def test_class_weights_scaling():
    """Verifies that class_weights (w_t^c) correctly scale the loss."""
    config = ModelConfig(
        task_structure=[[1]], 
        task_names=["task_binary"],
        task_types=["binary"],
        num_classes_per_task=[2]
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())
    
    y_true = {'task_binary': tf.constant([[1.0]], dtype=tf.float32)}
    y_pred = {'task_binary': tf.constant([[0.5]], dtype=tf.float32)}
    
    # Weight class 1 as 5x more important
    loss_fn = trainer._build_loss_function(
        num_tasks=1,
        class_weights=[{0: 1.0, 1: 5.0}],
        task_weights=[1.0]
    )
    
    loss = loss_fn(y_true, y_pred)
    # (0.693 * 5.0) / 1.0 = 3.465
    assert loss.numpy() == pytest.approx(3.465, abs=0.01)


def test_task_weights_importance():
    """Verifies task-level importance scaling (nu_k) for mixed corpora."""
    config = ModelConfig(
        task_structure=[[2]], 
        task_names=['task_binary', 'task_multiclass'],
        task_types=['binary', 'multiclass'],
        num_classes_per_task=[2, 3]
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())
    
    y_true = {
        'task_binary': tf.constant([[1.0]], dtype=tf.float32),
        'task_multiclass': tf.constant([0], dtype=tf.int32)
    }
    y_pred = {
        'task_binary': tf.constant([[0.5]], dtype=tf.float32),
        'task_multiclass': tf.constant([[0.5, 0.25, 0.25]], dtype=tf.float32)
    }
    
    # Prioritize task_binary by 10x
    loss_fn = trainer._build_loss_function(
        num_tasks=2,
        class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0, 2: 1.0}],
        task_weights=[10.0, 1.0] 
    )
    
    loss = loss_fn(y_true, y_pred)
    # Expected with multiclass one-vs-rest BCE:
    # binary(0.5, y=1)=0.693; multiclass sum BCE(label=0,[0.5,0.25,0.25])=1.269
    # total = (0.693 * 10) + (1.269 * 1) = 8.199
    assert loss.numpy() == pytest.approx(8.199, abs=0.05)


def test_task_type_weights_grouping_and_within_type_factor():
    config = ModelConfig(
        task_structure=[[3]],
        task_names=["task_a1", "task_a2", "task_b"],
        task_types=["binary", "binary", "binary"],
        num_classes_per_task=[2, 2, 2],
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())

    # D_A has 2 datapoints with any valid label across task_a1/task_a2.
    # D_B has 3 datapoints with valid labels for task_b.
    y_labels = {
        "task_a1": tf.constant([[1.0], [-1.0], [-1.0], [-1.0]], dtype=tf.float32),
        "task_a2": tf.constant([[-1.0], [1.0], [-1.0], [-1.0]], dtype=tf.float32),
        "task_b": tf.constant([[1.0], [1.0], [1.0], [-1.0]], dtype=tf.float32),
    }
    x_dummy = tf.zeros((4, 2), dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_labels)).batch(4)

    task_type_mapping = {"task_a1": "A", "task_a2": "A", "task_b": "B"}
    task_weights = trainer._compute_task_weights(ds, task_type_mapping=task_type_mapping)

    assert task_weights[0] == pytest.approx(1.2, abs=1e-4)
    assert task_weights[1] == pytest.approx(1.2, abs=1e-4)
    assert task_weights[2] == pytest.approx(0.8, abs=1e-4)

    y_true = {
        "task_a1": tf.constant([[1.0]], dtype=tf.float32),
        "task_a2": tf.constant([[1.0]], dtype=tf.float32),
        "task_b": tf.constant([[1.0]], dtype=tf.float32),
    }
    y_pred = {
        "task_a1": tf.constant([[0.5]], dtype=tf.float32),
        "task_a2": tf.constant([[0.5]], dtype=tf.float32),
        "task_b": tf.constant([[0.5]], dtype=tf.float32),
    }
    loss_fn = trainer._build_loss_function(
        num_tasks=3,
        class_weights=[{0: 1.0, 1: 1.0}] * 3,
        task_weights=task_weights,
        task_type_mapping=task_type_mapping,
    )
    loss = loss_fn(y_true, y_pred)

    # ν_A*(1/|T_A|)*(L_a1+L_a2) + ν_B*(1/|T_B|)*L_b = 1.2*0.693 + 0.8*0.693
    assert loss.numpy() == pytest.approx(1.386, abs=0.05)


def test_multiclass_bce_with_masking_and_class_weights():
    config = ModelConfig(
        task_structure=[[1]],
        task_names=["task_multiclass"],
        task_types=["multiclass"],
        num_classes_per_task=[3],
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())

    y_true = {"task_multiclass": tf.constant([-1, 2], dtype=tf.int32)}
    y_pred = {
        "task_multiclass": tf.constant(
            [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]], dtype=tf.float32
        )
    }
    loss_fn = trainer._build_loss_function(
        num_tasks=1,
        class_weights=[{0: 1.0, 1: 2.0, 2: 3.0}],
        task_weights=[1.0],
    )
    loss = loss_fn(y_true, y_pred)

    # First row masked. Second row contributes:
    # -log(0.9)*1 + -log(0.8)*2 + -log(0.7)*3 = 1.6217
    assert loss.numpy() == pytest.approx(1.6217, abs=0.02)


def test_regression_standardization_remains_active():
    config = ModelConfig(
        task_structure=[[1]],
        task_names=["task_regression"],
        task_types=["regression"],
        num_classes_per_task=[1],
    )
    model = MultiTaskModel(config=config, embedding_dim=128)
    trainer = Trainer(model=model, config=TrainingConfig())
    trainer.model._regression_stats = {"task_regression": {"mean": 10.0, "std": 2.0}}

    y_true = {"task_regression": tf.constant([[12.0]], dtype=tf.float32)}
    y_pred = {"task_regression": tf.constant([[10.0]], dtype=tf.float32)}
    loss_fn = trainer._build_loss_function(
        num_tasks=1,
        class_weights=[{0: 1.0}],
        task_weights=[1.0],
    )
    loss = loss_fn(y_true, y_pred)

    # ((12-10)/2)^2 = 1.0 with regression standardization applied.
    assert loss.numpy() == pytest.approx(1.0, abs=0.001)


def test_warmup_call_compatibility(dummy_trainer):
    """Verifies fix for WarmUp schedule call signature."""
    dummy_trainer.config.use_warmup = True
    dummy_trainer.config.use_lr_schedule = True
    lr_schedule = dummy_trainer._build_lr_schedule(steps_per_epoch=100)
    
    # Ensure the schedule can be called with a step (standard Keras)
    val = lr_schedule(tf.constant(10, dtype=tf.int64))
    assert val > 0
