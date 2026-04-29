"""
Trainer class for multi-task models.

This module provides the Trainer class that handles training loops, loss computation,
optimization, and threshold selection for multi-task learning models.
"""
import json
import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from sklearn.metrics import (
    roc_curve, f1_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from multitask.config import TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.utils.compat import WarmUpWrapper


class Trainer:
    """Trainer for multi-task learning models.
    
    Handles the complete training pipeline for multi-task learning including:
    - Multi-task loss computation: computes individual losses per task based on type,
      applies masking and weighting, and combines them for joint optimization
    - Learning rate scheduling with optional warmup
    - Early stopping on validation loss
    - Optional threshold optimization for binary/multilabel tasks on validation data
    - Distributed training support via TensorFlow strategies
    - Model evaluation with task-specific and overall metrics
    
    All components (loss function, optimizer, scheduler) can be customized by the user.
    
    Args:
        model: The MultiTaskModel to train
        config: TrainingConfig with hyperparameters
        strategy: TensorFlow distribution strategy (optional, for distributed training)
        loss_fn: Custom loss function (if None, uses built-in multi-task loss)
        optimizer: Custom optimizer (if None, builds from config)
        
    Example:
        >>> config = TrainingConfig(epochs=20, batch_size=32, learning_rate=3e-4)
        >>> trainer = Trainer(model, config)
        >>> history, _ = trainer.fit(train_dataset, val_dataset, num_tasks=10)
        >>> # With threshold optimization:
        >>> history, thresholds = trainer.fit(train_dataset, val_dataset, 
        ...                                    num_tasks=10, optimize_thresholds=True)
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        config: Optional[TrainingConfig] = None,
        strategy: Optional[tf.distribute.Strategy] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.strategy = strategy or tf.distribute.get_strategy()
        self.custom_loss_fn = loss_fn
        self.custom_optimizer = optimizer
        # Missing-label masking follows ModelConfig.mask_value (not TrainingConfig.mask_value).
        self.mask_value = self.model.config.mask_value
        
        # Will be set during training
        self.history = None
        self.optimal_thresholds = None
    
    def _validate_multilabel(self, y_true_dict, mask_value=None):
        """
        For multilabel tasks, either all labels are present per observation,
        or all labels are missing and then masked. Partial masking is not allowed.
        """
        if mask_value is None:
            mask_value = self.mask_value
            
        for task_name, y_task_true in y_true_dict.items():
            real_idx = self.model.config.task_names.index(task_name)
            task_type = self.model.config.get_task_type(real_idx)
            
            if task_type != 'multilabel':
                continue
            
            is_missing = tf.equal(y_task_true, mask_value)
            num_missing = tf.reduce_sum(tf.cast(is_missing, tf.int32), axis=1)
            num_labels = tf.shape(y_task_true)[-1]
            
            partial_mask = tf.logical_and(num_missing > 0, num_missing < num_labels)
            
            if tf.reduce_any(partial_mask):
                raise ValueError(f"Multilabel task '{task_name}' has partially missing observations, which is not allowed.\n"
                                 "Each observation must have either all labels present or all labels missing and masked.\n"
                                 "Instead, consider splitting the task into separate binary tasks.")
    
    def _compute_regression_stats(
        self,
        dataset: tf.data.Dataset,
        num_samples_to_check: int = 10000
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean and std for regression tasks from training data.
        
        Used to standardize regression targets in the loss function,
        preventing regression loss from dominating classification loss.
        
        Args:
            dataset: Training dataset to compute stats from
            num_samples_to_check: Max number of samples to scan
            
        Returns:
            Dict of {task_name: {"mean": float, "std": float}} for each
            regression task. Empty dict if no regression tasks or
            standardization is disabled.
        """
        if not self.model.config.standardize_regression:
            return {}
        
        task_names = self.model.task_names or [f'task_{i}' for i in range(self.model.config.total_outputs)]
        
        # Identify regression tasks
        regression_tasks = []
        for task_name in task_names:
            real_idx = list(self.model.task_names).index(task_name)
            if self.model.config.get_task_type(real_idx) == 'regression':
                regression_tasks.append(task_name)
        
        if not regression_tasks:
            return {}
        
        # Collect non-masked labels
        task_values = {name: [] for name in regression_tasks}
        total_samples = 0
        
        for inputs, labels_dict in dataset.take(num_samples_to_check):
            batch_size = tf.shape(list(labels_dict.values())[0])[0].numpy()
            total_samples += batch_size
            
            for task_name in regression_tasks:
                labels = labels_dict[task_name].numpy().ravel()
                # Filter mask_value AND NaN. Plain `!=` lets NaN through
                # (NaN != anything is True), corrupting mean/std with NaN.
                labels_float = labels.astype(np.float64)
                valid = labels[(labels_float != self.mask_value) & ~np.isnan(labels_float)]
                if len(valid) > 0:
                    task_values[task_name].extend(valid.tolist())
            
            if total_samples >= num_samples_to_check:
                break
        
        # Compute mean and std
        stats = {}
        for task_name in regression_tasks:
            values = np.array(task_values[task_name])
            if len(values) >= 2:
                stats[task_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            else:
                # Not enough data — use identity transform
                stats[task_name] = {"mean": 0.0, "std": 1.0}
        
        return stats
                
    def _build_loss_function(
        self,
        num_tasks: int,
        class_weights: List[Dict[int, float]],
        task_weights: List[float],
        mask_value: Optional[float] = None
    ) -> Callable:
        """Build multi-task loss function with task-specific losses.
        
        This function implements the core of multi-task learning: it computes
        individual losses for each task based on its type, applies masking and
        weighting, then combines them into a single scalar loss for optimization.
        
        The loss function handles different loss types per task:
        - Binary: binary crossentropy with sigmoid outputs
        - Multiclass: categorical crossentropy with softmax outputs
        - Multilabel: binary crossentropy with sigmoid outputs (one per class)
        
        Masking: Labels with value mask_value (-1 by default) are excluded from
        both loss computation and normalization, enabling handling of missing labels.
        
        Task weighting: Each task can have importance weight applied to its loss,
        allowing prioritization of certain tasks during training. Weights are
        automatically computed based on task sample sizes (inversely proportional).
        
        Class weighting: Each class within a task can have importance weight applied,
        handling class imbalance. Weights are automatically computed based on class
        frequencies (inversely proportional).
        
        Args:
            num_tasks: Total number of output tasks
            class_weights: Per-task class weights, list of {0: weight, 1: weight} dicts
                          (automatically computed in fit() if not provided by user)
            task_weights: Relative importance weights for each task
                         (automatically computed in fit() if not provided by user)
            mask_value: Value indicating missing labels to be excluded (default: -1)
            
        Returns:
            Loss function that takes (y_true_dict, y_pred_dict) and returns scalar loss
        """
        if mask_value is None:
            mask_value = self.mask_value
            
        task_weights_tf = tf.cast(task_weights, tf.float32)
        
        def dict_weighted_masked_loss(y_true_dict: Any, y_pred_dict: Any):
            """Compute multi-task loss combining individual task losses.
            
            For each task:
            1. Compute task-specific loss based on task type (binary/multiclass/multilabel)
            2. Apply mask to exclude invalid labels
            3. Apply task weight to prioritize certain tasks
            4. Accumulate into total loss
            
            Final loss is averaged over all valid (non-masked) samples.
            
            Note: During graph construction, Keras may pass SymbolicTensor objects
            instead of dicts. This function handles both cases.
            """
            # Convert to dict if not already (handles SymbolicTensor case during graph construction)
            try:
                # Try to access as dict - if this works, it's already a dict
                _ = y_true_dict.keys()
            except (AttributeError, TypeError):
                # Not a dict - try to convert using model structure
                if self.model.task_names:
                    task_names_list = sorted(self.model.task_names)
                else:
                    task_names_list = [f'task_{i}' for i in range(num_tasks)]
                
                # Try to convert: assume it's a list/tuple of tensors matching task order
                try:
                    if hasattr(y_true_dict, '__len__') and not isinstance(y_true_dict, (str, bytes)):
                        # Convert list/tuple to dict
                        y_true_dict = {task_names_list[i]: y_true_dict[i] for i in range(len(task_names_list))}
                        y_pred_dict = {task_names_list[i]: y_pred_dict[i] for i in range(len(task_names_list))}
                    else:
                        # Can't convert - raise informative error
                        raise TypeError(
                            f"Cannot convert y_true_dict from {type(y_true_dict).__name__} to dict."
                        )
                except (TypeError, IndexError, AttributeError) as e:
                    raise TypeError(
                        f"Cannot convert y_true_dict from {type(y_true_dict).__name__} to dict."
                        f"Original error: {e}"
                    )
            
            total_loss = 0.0
            num_valid_samples = 0
            
            for real_idx, task_name in enumerate(self.model.task_names):
                y_task_true = y_true_dict[task_name]
                y_task_pred = y_pred_dict[task_name]
                task_type = self.model.config.get_task_type(real_idx)
                
                if task_type == 'regression':
                    # MSE loss, no class weighting
                    y_true_flat = tf.reshape(y_task_true, [-1])
                    y_pred_flat = tf.reshape(y_task_pred, [-1])
                    mask = tf.cast(
                        tf.logical_and(
                            tf.not_equal(y_true_flat, tf.cast(mask_value, y_true_flat.dtype)),
                            tf.logical_not(tf.math.is_nan(y_true_flat))), tf.float32)
                    # Zero out masked/NaN positions to prevent NaN propagation
                    # through tf.square below; the mask cancels their contribution.
                    y_true_flat = tf.where(
                        tf.cast(mask, tf.bool),
                        y_true_flat,
                        tf.zeros_like(y_true_flat),
                    )
                    
                    # Standardize both target and prediction before MSE.
                    # This is equivalent to MSE / std², which scales down the
                    # regression loss to be comparable with classification losses.
                    # The model still predicts in original scale
                    # transform needed at inference.
                    
                    if (hasattr(self.model, '_regression_stats')
                            and task_name in self.model._regression_stats):
                        stats = self.model._regression_stats[task_name]
                        mean = tf.constant(stats['mean'], dtype=tf.float32)
                        std = tf.constant(stats['std'] + 1e-7, dtype=tf.float32)
                        y_true_flat = (y_true_flat - mean) / std
                        y_pred_flat = (y_pred_flat - mean) / std
                    
                    loss_val = tf.square(y_true_flat - y_pred_flat)
                
                elif task_type == 'multiclass':
                    # Masked labels must not be passed to sparse CE (-1 is an
                    # invalid class index).
                    # Use a safe dummy class (0) where masked; those positions are
                    # zeroed by mask.
                    y_true_flat = tf.cast(tf.reshape(y_task_true, [-1]), tf.int32)
                    mask_val_int = tf.cast(mask_value, y_true_flat.dtype)
                    mask = tf.cast(tf.not_equal(y_true_flat, mask_val_int),
                        tf.float32)
                    safe_labels = tf.where(
                        tf.equal(y_true_flat, mask_val_int),
                        tf.zeros_like(y_true_flat),
                        y_true_flat,
                    )
                    loss_val = tf.keras.losses.sparse_categorical_crossentropy(
                        safe_labels, y_task_pred, from_logits=False
                    )
                    loss_val = tf.reshape(loss_val, [-1])
                    num_classes = self.model.config.get_num_classes_per_task(real_idx)
                    weights_list = [
                        float(class_weights[real_idx].get(i, 1.0)) for i in range(num_classes)
                    ]
                    weights_map = tf.constant(weights_list, dtype=tf.float32)
                    sample_class_weights = tf.gather(weights_map, safe_labels)
                    loss_val = loss_val * sample_class_weights
                else:
                    y_true_flat = tf.reshape(y_task_true, [-1])
                    y_pred_flat = tf.reshape(y_task_pred, [-1])
                    mask_val = tf.cast(mask_value, y_true_flat.dtype)
                    mask = tf.cast(
                        tf.logical_and(
                            tf.not_equal(y_true_flat, mask_val),
                            tf.logical_not(tf.math.is_nan(y_true_flat))), tf.float32)
                    # Replace ALL invalid positions (mask_value AND NaN) with 0
                    # to prevent NaN from propagating through binary_crossentropy;
                    # the mask zeros out their contribution downstream.
                    safe_labels = tf.where(
                        tf.cast(mask, tf.bool),
                        y_true_flat,
                        tf.zeros_like(y_true_flat),
                    )
                    loss_val = tf.keras.losses.binary_crossentropy(
                        tf.reshape(safe_labels, [-1, 1]),
                        tf.reshape(y_pred_flat, [-1, 1]),
                    )
                    loss_val = tf.reshape(loss_val, [-1])
                
                # Class weighting: skip for regression
                if task_type == 'regression':
                    pass  # no class weighting
                
                elif task_type in ['binary', 'multilabel']:
                    cw0 = tf.cast(class_weights[real_idx].get(0, 1.0), tf.float32)
                    cw1 = tf.cast(class_weights[real_idx].get(1, 1.0), tf.float32)
                    sample_weights = tf.where(tf.equal(y_true_flat, 0), cw0, cw1)
                    loss_val = loss_val * sample_weights
                
                elif task_type == 'multiclass':
                    pass
                    
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
                
                # final aggregation and normalization
                weighted_loss = loss_val * mask * task_weights_tf[real_idx]
                total_loss += tf.reduce_sum(weighted_loss)
                num_valid_samples += tf.reduce_sum(mask)
            
            # Average over valid entries
            return total_loss / (num_valid_samples + 1e-7)
        
        return dict_weighted_masked_loss
    
    def _build_lr_schedule(
        self,
        steps_per_epoch: int
    ) -> Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Build learning rate schedule.
        
        Args:
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            Learning rate or learning rate schedule
        """
        if not self.config.use_lr_schedule:
            return self.config.learning_rate
        
        num_train_steps = steps_per_epoch * self.config.epochs
        
        # Build decay schedule
        if self.config.lr_schedule_type == 'polynomial_decay':
            schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=num_train_steps,
                end_learning_rate=0.0
            )
        elif self.config.lr_schedule_type == 'exponential_decay':
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=0.96
            )
        elif self.config.lr_schedule_type == 'cosine_decay':
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=num_train_steps
            )
        elif self.config.lr_schedule_type == 'constant':
            return self.config.learning_rate
        else:
            raise ValueError(f"Unknown lr_schedule_type: {self.config.lr_schedule_type}")
        
        # Add warmup if enabled
        # fix: use native keras 3 compatible warmup wrapper instead of official.nlp optimization.WarmUp
        if self.config.use_warmup:
            num_warmup_steps = max(1, int(num_train_steps * self.config.warmup_percent / 100))
            
            return WarmUpWrapper(
                warmup_steps=num_warmup_steps,
                decay_schedule_fn=schedule,
                initial_learning_rate=self.config.learning_rate
            )
        
        return schedule
    
    def _build_optimizer(
        self,
        steps_per_epoch: int
    ) -> tf.keras.optimizers.Optimizer:
        """Build optimizer with learning rate schedule.
        
        Args:
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            Configured optimizer
        """
        lr_schedule = self._build_lr_schedule(steps_per_epoch)
        opt_type = self.config.optimizer_type.lower()
        
        # common optimizers
        optimizer_map = {
            'adamw': tf.keras.optimizers.AdamW,
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'nadam': tf.keras.optimizers.Nadam
        }
        
        optimizer_kwargs = getattr(self.config, 'optimizer_kwargs', {}) or {}
        
        if opt_type in optimizer_map:
            optimizer = optimizer_map[opt_type](
                learning_rate=lr_schedule,
                **optimizer_kwargs
            )
        else:
            # get optimizer from keras registry for additional optimizers
            try:
                optimizer = tf.keras.optimizers.get(opt_type)
            except ValueError:
                raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")
            
            for key, val in optimizer_kwargs.items():
                if hasattr(optimizer, key):
                    setattr(optimizer, key, val)
                else:
                    raise ValueError(f"Optimizer {opt_type} does not have attribute '{key}' for optimizer_kwargs")
        
        return optimizer
    
    def _derive_pooled_labels(
        self,
        y_true_dict: Dict[str, np.ndarray],
        mask_value: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Derive ground-truth labels for pooled outputs from task labels.
        
        Applies the same pooling strategy to ground-truth labels as the model
        applies to predictions.  Handles masking: if ALL labels for
        an observation are masked, the pooled label is masked; otherwise the
        pool is computed over non-masked constituents only.
        
        Args:
            y_true_dict: Dictionary of true labels {task_name: array} — task-level
                         only (no pool keys expected).
            mask_value: Value indicating missing labels (default: -1).
            
        Returns:
            Dictionary of {pool_name: pooled_labels_array} for each configured pool.
            Empty dict if no pools are configured.
        """
        if not self.model.config.has_pools:
            return {}

        if mask_value is None:
            mask_value = self.mask_value

        pooled_labels = {}
        for pool_name in self.model.config.pool_names:
            pool_cfg = self.model.config.get_pool_config(pool_name)
            strategy = pool_cfg['strategy']
            constituent_tasks = pool_cfg['tasks']
            
            # Gather constituent label arrays.
            # Each is shape [n_samples] or [n_samples, 1] for binary tasks.
            constituent_arrays = []
            for t in constituent_tasks:
                arr = y_true_dict[t]
                # Flatten to 1D if [n_samples, 1]
                if arr.ndim > 1 and arr.shape[-1] == 1:
                    arr = arr.ravel()
                constituent_arrays.append(arr)
            
            # Stack: shape [n_samples, n_tasks]
            stacked = np.stack(constituent_arrays, axis=1)
            
            # Build mask: True where the label is NOT masked
            valid = (stacked != mask_value)
            
            # If ALL constituents are masked for an observation, the pool is masked.
            all_masked = ~valid.any(axis=1)  # [n_samples]
            
            # Replace masked entries with NaN so they don't affect the pool op,
            # then apply the strategy over axis=1.
            stacked_float = stacked.astype(np.float64)
            stacked_float[~valid] = np.nan
            
            with np.errstate(all='ignore'):  # suppress nanmean warnings for all-NaN slices
                if strategy == 'max':
                    pooled = np.nanmax(stacked_float, axis=1)
                elif strategy == 'mean':
                    pooled = np.nanmean(stacked_float, axis=1)
                elif strategy == 'min':
                    pooled = np.nanmin(stacked_float, axis=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {strategy}")
            
            # Restore mask value for fully-masked observations
            pooled[all_masked] = mask_value
            
            # For binary pool ground truth with 'mean' strategy, the result is
            # fractional (e.g., 0.33 if 1 of 3 task is positive).
            # Threshold optimization will handle this — no rounding here.
            
            pooled_labels[pool_name] = pooled.astype(np.float32)
        
        return pooled_labels
    
    def _find_optimal_thresholds(
        self,
        y_true_dict: Dict[str, np.ndarray],
        y_pred_dict: Dict[str, np.ndarray],
        mask_value: Optional[float] = None
    ) -> Tuple[Dict[str, Union[float, List[float], None]], Dict[str, float]]:
        """Find optimal classification thresholds using validation data.
        
        Uses Youden's J statistic (max sensitivity + specificity) to find
        the optimal threshold for each binary/multilabel task independently.
        Multiclass tasks don't use thresholds (use argmax instead).
        
        Handles both architectural task names and pooled output names.
        
        Args:
            y_true_dict: Dictionary of true labels {name: [n_samples, ...]}
                         May contain both task names and pool names.
            y_pred_dict: Dictionary of predicted probabilities {name: [n_samples, ...]}
            mask_value: Value indicating missing labels
            
        Returns:
            Tuple of (thresholds_dict, f1_scores_dict)
            where name -> float for binary, List[float] for multilabel, 
            None for multiclass/regression
        """
        if mask_value is None:
            mask_value = self.mask_value
            
        thresholds = {}
        f1_scores = {}
        
        output_names = sorted(y_true_dict.keys())
        
        for name in output_names:
            # Look up task type: pool or architectural task
            if self.model.config.is_pool(name):
                task_type = self.model.config.get_pool_task_type(name)
            else:
                task_idx = list(self.model.task_names).index(name) if self.model.task_names else int(name.split('_')[-1])
                task_type = self.model.config.get_task_type(task_idx)
            
            y_task_true = y_true_dict[name]
            y_task_pred = y_pred_dict[name]
            
            if task_type == 'multiclass':
                # Multiclass use argmax
                thresholds[name] = None
                f1_scores[name] = 0.0
            
            elif task_type == 'regression':
                thresholds[name] = None
                f1_scores[name] = None
                continue

            elif task_type == 'binary':
                if y_task_pred.ndim > 1:
                    y_task_pred = y_task_pred[:, 0]
                    y_task_true = y_task_true[:, 0] if y_task_true.ndim > 1 else y_task_true
                
                valid_mask = np.isfinite(y_task_true) & (y_task_true != mask_value)
                y_task_true_valid = y_task_true[valid_mask]
                y_task_pred_valid = y_task_pred[valid_mask]
                
                if len(y_task_true_valid) == 0:
                    thresholds[name] = 0.5
                    f1_scores[name] = 0.0
                    continue
                
                if len(np.unique(y_task_true_valid)) < 2:
                    thresholds[name] = 0.5
                    f1_scores[name] = 0.0
                    continue
                    
                fpr, tpr, thresh = roc_curve(y_task_true_valid, y_task_pred_valid)
                optimal_threshold = thresh[np.argmax(tpr - fpr)]
                
                y_pred_binary = (y_task_pred_valid > optimal_threshold).astype(int)
                thresholds[name] = float(optimal_threshold)
                f1_scores[name] = float(f1_score(y_task_true_valid, y_pred_binary, average='weighted', zero_division=0))

            elif task_type == 'multilabel':
                num_labels = y_task_pred.shape[1]
                label_thresholds = []
                label_f1s = []
                
                for col_idx in range(num_labels):
                    y_col_true = y_task_true[:, col_idx] if y_task_true.ndim > 1 else y_task_true
                    y_col_pred = y_task_pred[:, col_idx]
                    
                    valid_mask = np.isfinite(y_col_true) & (y_col_true != mask_value)
                    y_col_true_valid = y_col_true[valid_mask]
                    y_col_pred_valid = y_col_pred[valid_mask]
                    
                    if len(y_col_true_valid) == 0:
                        label_thresholds.append(0.5)
                        label_f1s.append(0.0)
                        continue
                    
                    if len(np.unique(y_col_true_valid)) < 2:
                        label_thresholds.append(0.5)
                        label_f1s.append(0.0)
                        continue
                        
                    fpr, tpr, thresh = roc_curve(y_col_true_valid, y_col_pred_valid)
                    opt_thresh = thresh[np.argmax(tpr - fpr)]
                    
                    y_col_binary = (y_col_pred_valid > opt_thresh).astype(int)
                    f1 = f1_score(y_col_true_valid, y_col_binary, average='binary', zero_division=0)
                    
                    label_thresholds.append(float(opt_thresh))
                    label_f1s.append(float(f1))
                
                # Store the entire array of thresholds for multilabel tasks
                thresholds[name] = label_thresholds
                f1_scores[name] = float(np.mean(label_f1s))
        
        return thresholds, f1_scores
    
    def _compute_task_weights(
        self,
        dataset: tf.data.Dataset,
        num_samples_to_check: int = 10000
    ) -> List[float]:
        """Compute task weights inversely proportional to non-masked sample count.
        
        Implements: ν_k ~ 1/|D_k| where D_k is the set of non-masked samples for task k.
        
        Args:
            dataset: Training dataset to compute weights from
            num_samples_to_check: Max number of samples to scan
            
        Returns:
            List of task weights (one per task)
        """
        task_names = self.model.task_names or [f'task_{i}' for i in range(self.model.config.total_outputs)]
        task_sample_counts = {name: 0 for name in task_names}
        total_samples = 0
        
        # Count non-masked samples per task
        for inputs, labels_dict in dataset.take(num_samples_to_check):
            batch_size = tf.shape(list(labels_dict.values())[0])[0].numpy()
            total_samples += batch_size
            
            for task_name in task_names:
                labels = labels_dict[task_name]
                # Count non-masked samples (not equal to self.mask_value)
                labels_float = tf.cast(labels, tf.float32) if not labels.dtype.is_floating else labels
                mask_val = tf.cast(self.mask_value, labels_float.dtype)
                non_masked = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.not_equal(labels_float, mask_val),
                            tf.logical_not(tf.math.is_nan(labels_float))
                        ),
                        tf.int32
                    )
                ).numpy()
                task_sample_counts[task_name] += non_masked
            
            if total_samples >= num_samples_to_check:
                break
        
        # Compute weights as inverse of sample counts, normalized
        task_weights = []
        for task_name in sorted(task_names):
            count = max(task_sample_counts[task_name], 1)  # Avoid division by zero
            task_weights.append(1.0 / count)
        
        # Normalize so weights sum to number of tasks (maintains scale)
        total_weight = sum(task_weights)
        task_weights = [w * len(task_weights) / total_weight for w in task_weights]
        
        return task_weights
    
    def _compute_class_weights(
        self,
        dataset: tf.data.Dataset,
        num_samples_to_check: int = 10000
    ) -> List[Dict[int, float]]:
        """Compute per-task class weights inversely proportional to class frequency.
        
        Implements: w_k^c proportional to inverse enrichment of class c in task k.
        
        Args:
            dataset: Training dataset to compute weights from
            num_samples_to_check: Max number of samples to scan
            
        Returns:
            List of dicts mapping class index to weight (one dict per task)
        """
        task_names = self.model.task_names or [f'task_{i}' for i in range(self.model.config.total_outputs)]
        
        # Collect labels for each task
        task_labels = {name: [] for name in task_names}
        total_samples = 0
        
        for inputs, labels_dict in dataset.take(num_samples_to_check):
            batch_size = tf.shape(list(labels_dict.values())[0])[0].numpy()
            total_samples += batch_size
            
            for task_name in sorted(task_names):
                labels = labels_dict[task_name].numpy()
                real_idx = list(self.model.task_names).index(task_name)
                task_type = self.model.config.get_task_type(real_idx)
                
                # Flatten and filter mask_value AND NaN. Plain `!=` lets
                # NaN through (NaN != anything is True), and labels.astype(int)
                # below converts NaN to a huge negative integer, crashing
                # np.bincount with "negative elements".
                if len(labels.shape) > 1:
                    labels = labels.flatten()
                labels_float = labels.astype(np.float64)
                valid_labels = labels[(labels_float != self.mask_value) & ~np.isnan(labels_float)]
                
                if len(valid_labels) > 0:
                    task_labels[task_name].extend(valid_labels.tolist())
            
            if total_samples >= num_samples_to_check:
                break
        
        # Compute class weights for each task
        class_weights_list = []
        for task_name in sorted(task_names):
            labels = np.array(task_labels[task_name])
            real_idx = list(self.model.task_names).index(task_name)
            task_type = self.model.config.get_task_type(real_idx)
            num_classes = self.model.config.get_num_classes_per_task(real_idx)
            
            # Regression tasks have no classes
            if task_type == 'regression':
                class_weights_list.append({0: 1.0})
                continue
            
            if len(labels) == 0:
                # No valid labels - use uniform weights
                if task_type == 'binary':
                    class_weights_list.append({0: 1.0, 1: 1.0})
                else:
                    class_weights_list.append({i: 1.0 for i in range(num_classes)})
                continue
            
            # Count class frequencies
            if task_type == 'binary':
                class_counts = np.bincount(labels.astype(int), minlength=2)
                if len(class_counts) < 2:
                    class_counts = np.array([class_counts[0] if len(class_counts) > 0 else 0, 0])
            elif task_type == 'multiclass':
                class_counts = np.bincount(labels.astype(int), minlength=num_classes)
            elif task_type == 'multilabel':
                # For multilabel, treat each class independently (binary per class)
                class_counts = np.bincount(labels.astype(int), minlength=2)
                if len(class_counts) < 2:
                    class_counts = np.array([class_counts[0] if len(class_counts) > 0 else 0, 0])
            
            # Compute weights as inverse frequency
            class_weights = {}
            for class_idx in range(len(class_counts)):
                count = class_counts[class_idx]
                if count == 0:
                    # Class not present - use very small weight to avoid division by zero
                    class_weights[class_idx] = 0.001
                else:
                    # Inverse frequency
                    class_weights[class_idx] = len(labels) / (len(class_counts) * count)
            
            class_weights_list.append(class_weights)
        
        return class_weights_list
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        num_tasks: int,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        class_weights: Optional[List[Dict[int, float]]] = None,
        task_weights: Optional[List[float]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        optimize_thresholds: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_monitor: str = 'val_loss',
        checkpoint_mode: str = 'min',
        verbose: int = 1
    ) -> Tuple[tf.keras.callbacks.History, Dict[str, Optional[float]]]:
        """Train the multi-task model.
        
        The training process:
        1. Compiles model with custom multi-task loss function
        2. Trains on provided dataset with optional early stopping
        3. Optionally optimizes classification thresholds on validation data
        
        The multi-task loss function computes individual losses for each task,
        applies masking and weighting, and combines them for joint optimization.
        
        Weights are automatically computed from training data if not provided:
        - Task weights: inversely proportional to non-masked sample count per task
        - Class weights: inversely proportional to class frequency within each task
        
        Args:
            train_dataset: Training TensorFlow Dataset yielding (inputs, labels_dict)
            val_dataset: Validation TensorFlow Dataset yielding (inputs, labels_dict)
            num_tasks: Total number of output tasks
            steps_per_epoch: Number of steps per epoch (auto-calculated if None)
            validation_steps: Number of validation steps (auto-calculated if None)
            class_weights: Optional per-task class weights for loss function.
                          If None, automatically computed from training data using
                          inverse class frequency weighting.
            task_weights: Optional task importance weights for relative task prioritization.
                         If None, automatically computed from training data using
                         inverse sample count weighting.
            callbacks: Additional Keras callbacks
            optimize_thresholds: Whether to find optimal classification thresholds
                                on validation data (only for binary/multilabel tasks).
                                If False, returns default thresholds (0.5 for binary/multilabel,
                                None for multiclass).
            checkpoint_dir: Optional directory to save model checkpoints during training.
                           If provided, automatically saves best model based on checkpoint_monitor.
            checkpoint_monitor: Metric to monitor for checkpointing (default: 'val_loss')
            checkpoint_mode: Whether to minimize or maximize checkpoint_monitor ('min' or 'max')
            verbose: Verbosity mode for model.fit() (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Tuple of (training history, thresholds dict):
            - history: Keras History object with training metrics
            - thresholds: Dict mapping task_name -> threshold value
                         (0.5 for binary/multilabel by default, None for multiclass)
                         If optimize_thresholds=True, contains optimized values
                         
        Example:
            >>> # Basic training
            >>> history, thresholds = trainer.fit(train_ds, val_ds, num_tasks=10)
            >>> 
            >>> # With threshold optimization and checkpointing
            >>> history, thresholds = trainer.fit(
            ...     train_ds, val_ds, num_tasks=10,
            ...     optimize_thresholds=True,
            ...     checkpoint_dir='checkpoints/model_v1',
            ...     checkpoint_monitor='val_loss'
            ... )
        """
        with self.strategy.scope():
            # Build or use custom loss function
            # This function encapsulates true multi-task learning: computing and combining
            # individual task losses with automatic task and class weighting
            if self.custom_loss_fn is not None:
                loss_fn = self.custom_loss_fn
            else:
                # Automatically compute weights if not provided
                if class_weights is None:
                    print("Computing class weights from training data...")
                    class_weights = self._compute_class_weights(train_dataset)
                    print(f"Computed class weights for {len(class_weights)} tasks")
                
                if task_weights is None:
                    print("Computing task weights from training data...")
                    task_weights = self._compute_task_weights(train_dataset)
                    print(f"Task weights: {[f'{w:.3f}' for w in task_weights]}")
                    
                if task_weights is None:
                    print("Computing task weights from training data...")
                    task_weights = self._compute_task_weights(train_dataset)
                    print(f"Task weights: {[f'{w:.3f}' for w in task_weights]}")
                
                # Compute regression standardization stats from training data
                regression_stats = self._compute_regression_stats(train_dataset)
                if regression_stats:
                    print("Regression target standardization:")
                    for name, s in regression_stats.items():
                        print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}")
                self.model._regression_stats = regression_stats
                
                loss_fn = self._build_loss_function(
                    num_tasks=num_tasks,
                    class_weights=class_weights,
                    task_weights=task_weights
                )
            
            # Determine steps per epoch
            if steps_per_epoch is None:
                # Try to infer from dataset cardinality
                steps_per_epoch = int(train_dataset.cardinality())
                if steps_per_epoch < 0:
                    raise ValueError("Could not determine steps_per_epoch automatically. Please provide it explicitly.")
            
            # Build or use custom optimizer
            if self.custom_optimizer is not None:
                optimizer = self.custom_optimizer
            else:
                optimizer = self._build_optimizer(steps_per_epoch)
            
            # Compile model with multi-task loss function
            # The loss function internally computes task-specific losses and combines them
            # for joint optimization, implementing the core of multi-task learning
            self.model._mtl_loss_fn = loss_fn
            
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                jit_compile=False
            )
            
            # Setup callbacks
            callback_list = callbacks or []
            
            if self.config.early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=self.config.early_stopping_min_delta,
                    patience=self.config.early_stopping_patience,
                    mode='min',
                    restore_best_weights=True,
                    verbose=1
                )
                callback_list.append(early_stop)
            
            # Add checkpointing if requested
            if checkpoint_dir is not None:
                # Create a custom callback that uses our save_checkpoint method
                class CheckpointCallback(tf.keras.callbacks.Callback):
                    def __init__(self, trainer, filepath, monitor='val_loss', mode='min'):
                        super().__init__()
                        self.trainer = trainer
                        self.filepath = filepath
                        self.monitor = monitor
                        self.mode = mode
                        self.best_value = float('inf') if mode == 'min' else float('-inf')
                        
                    def on_epoch_end(self, epoch, logs=None):
                        current = logs.get(self.monitor)
                        if current is None:
                            return
                        
                        is_better = (
                            (self.mode == 'min' and current < self.best_value) or
                            (self.mode == 'max' and current > self.best_value)
                        )
                        
                        if is_better:
                            self.best_value = current
                            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saving checkpoint...")
                            self.trainer.save_checkpoint(self.filepath, save_format='tf')
                
                checkpoint_callback = CheckpointCallback(
                    trainer=self,
                    filepath=checkpoint_dir,
                    monitor=checkpoint_monitor,
                    mode=checkpoint_mode
                )
                callback_list.append(checkpoint_callback)
            
            # Train model
            print(f"\nTraining with config:")
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  Weight decay: {self.config.weight_decay_rate}")
            print(f"  Dropout rate: {self.config.dropout_rate}")
            print(f"  Steps per epoch: {steps_per_epoch}\n")
            
            self.history = self.model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callback_list,
                verbose=verbose
            )
            
            # Optionally find optimal thresholds on validation data
            if optimize_thresholds:
                print("\nFinding optimal classification thresholds...")
                y_val_pred_dict = self.model.predict(val_dataset, verbose=0)
                
                # Extract true labels from validation dataset as dictionary.
                # The dataset only contains architectural task labels — pool
                # labels will be derived below.
                task_names_list = self.model.task_names or [f'task_{i}' for i in range(self.model.config.total_outputs)]
                y_val_true_dict = {name: [] for name in task_names_list}
                
                for _, labels_dict in val_dataset:
                    for task_name in task_names_list:
                        y_val_true_dict[task_name].append(labels_dict[task_name].numpy())
                
                for task_name in task_names_list:
                    y_val_true_dict[task_name] = np.concatenate(y_val_true_dict[task_name], axis=0)
                
                # Derive pooled ground-truth labels from task labels
                pooled_true = self._derive_pooled_labels(y_val_true_dict)
                y_val_true_dict.update(pooled_true)
                
                self.optimal_thresholds, f1_scores = self._find_optimal_thresholds(
                    y_val_true_dict, y_val_pred_dict
                )
                
                valid_f1_scores = [f1 for f1 in f1_scores.values() if f1 is not None]
                if valid_f1_scores:
                    n_zero = sum(1 for f1 in valid_f1_scores if f1 == 0.0)
                    suffix = (f" ({n_zero}/{len(valid_f1_scores)} tasks scored 0)"
                              if n_zero else f" (n={len(valid_f1_scores)})")
                    print(f"Average validation F1: {np.mean(valid_f1_scores):.4f}{suffix}")
            else:
                # Use default thresholds for transparency
                print("\nUsing default thresholds (0.5 for binary/multilabel, None for multiclass)")
                self.optimal_thresholds = {}
                task_names = self.model.task_names or [f'task_{i}' for i in range(num_tasks)]
                
                for task_name in sorted(task_names):
                    real_idx = list(self.model.task_names).index(task_name)
                    task_type = self.model.config.get_task_type(real_idx)
                    if task_type in ('multiclass', 'regression'):
                        self.optimal_thresholds[task_name] = None
                    elif task_type == 'binary':
                        self.optimal_thresholds[task_name] = 0.5
                    elif task_type == 'multilabel':
                        num_labels = self.model.config.get_num_classes_per_task(real_idx)
                        self.optimal_thresholds[task_name] = [0.5] * num_labels
                
                # Add default thresholds for pools
                for pool_name in self.model.config.pool_names:
                    pool_type = self.model.config.get_pool_task_type(pool_name)
                    if pool_type in ('multiclass', 'regression'):
                        self.optimal_thresholds[pool_name] = None
                    elif pool_type == 'binary':
                        self.optimal_thresholds[pool_name] = 0.5
                    elif pool_type == 'multilabel':
                        num_labels = self.model.config.get_pool_num_classes(pool_name)
                        self.optimal_thresholds[pool_name] = [0.5] * num_labels
            
            # Always return tuple for consistency
            return self.history, self.optimal_thresholds
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        thresholds: Optional[Dict[str, Union[float, List[float], None]]] = None,
        metrics: Optional[List[str]] = None,
        mask_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset.
        
        Computes specified metrics on the provided dataset. Handles mixed task types
        (binary, multiclass, multilabel) and pooled outputs. Can use stored thresholds
        from training (if optimize_thresholds=True was used in fit()), custom
        thresholds, or compute probability-based metrics only.
        
        Threshold Usage:
            - If thresholds is None and self.optimal_thresholds exists: uses stored thresholds
            - If thresholds is provided: uses those thresholds (dict format)
            - If neither: only computes probability-based metrics (AUC, AP)
            - Multiclass tasks ignore thresholds (use argmax)
        
        Args:
            dataset: TensorFlow Dataset to evaluate on (yields (inputs, labels_dict))
            thresholds: Optional dict of thresholds per output name (tasks and pools).
                       Values may be float or None (e.g. multiclass/regression).
                       If None, will use self.optimal_thresholds if available
            metrics: List of metric names to compute. Options:
                     Binary metrics (require thresholds): 'accuracy', 'precision', 'recall', 'f1'
                     Probability metrics (no threshold): 'auc', 'average_precision'
                     If None, computes all applicable metrics based on threshold availability
            mask_value: Value indicating missing labels to mask out
            
        Returns:
            Dictionary containing:
                - 'probabilities_dict': Dict of predicted probabilities {name: [n_samples, n_classes]}
                - 'predictions_dict': Dict of binary predictions (if thresholds provided)
                - 'task_metrics': Per-task metrics dict {task_name: {metric: value}}
                - 'pool_metrics': Per-pool metrics dict {pool_name: {metric: value}} (if pools configured)
                - 'overall_metrics': Overall metrics dict {metric: average_value} (tasks only)
                - 'thresholds_used': Dict of thresholds used (if applicable)
                
        Example:
            >>> # After training with optimize_thresholds=True
            >>> results = trainer.evaluate(test_dataset)  # Uses stored thresholds
            >>> 
            >>> # With custom thresholds
            >>> results = trainer.evaluate(test_dataset, thresholds={'task_0': 0.6, ...})
            >>> 
            >>> # Probability-based metrics only
            >>> results = trainer.evaluate(test_dataset, metrics=['auc', 'average_precision'])
        """
        if mask_value is None:
            mask_value = self.mask_value
        
        # Get predictions (includes both task and pool outputs)
        print("Generating predictions...")
        y_pred_dict = self.model.predict(dataset, verbose=0)
        
        # Extract true labels for architectural tasks only from dataset.
        # The dataset does not contain pool labels — those are derived.
        task_names_list = self.model.task_names or [f'task_{i}' for i in range(self.model.config.total_outputs)]
        y_true_dict = {name: [] for name in task_names_list}
        
        for _, labels_dict in dataset:
            for task_name in task_names_list:
                y_true_dict[task_name].append(labels_dict[task_name].numpy())
        
        for task_name in task_names_list:
            y_true_dict[task_name] = np.concatenate(y_true_dict[task_name], axis=0)
        
        # Derive pooled ground-truth labels
        pooled_true = self._derive_pooled_labels(y_true_dict, mask_value=mask_value)
        y_true_dict.update(pooled_true)
        
        # All output names: tasks + pools
        all_output_names = sorted(y_pred_dict.keys())
        
        # Determine which thresholds to use
        use_thresholds = thresholds if thresholds is not None else self.optimal_thresholds
        
        # Determine which metrics to compute
        if metrics is None:
            if use_thresholds is not None:
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']
            else:
                metrics = ['auc', 'average_precision']
        
        # Validate metrics
        classification_metrics = {'accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision'}
        regression_metrics = {'mse', 'mae', 'r2', 'pearson_r'}
        valid_metrics = classification_metrics | regression_metrics
        
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Unknown metric: {metric}. Valid options: {valid_metrics}")
        
        # Apply thresholds if available to create binary predictions
        y_pred_binary_dict = None
        if use_thresholds is not None:
            y_pred_binary_dict = {}
            for name in all_output_names:
                # Look up task type
                if self.model.config.is_pool(name):
                    task_type = self.model.config.get_pool_task_type(name)
                else:
                    task_idx = list(self.model.task_names).index(name) if self.model.task_names else all_output_names.index(name)
                    task_type = self.model.config.get_task_type(task_idx)
                
                if task_type == 'regression':
                    # Regression: store raw predictions
                    y_pred_binary_dict[name] = y_pred_dict[name].ravel()
                elif task_type == 'multiclass':
                    y_pred_binary_dict[name] = np.argmax(y_pred_dict[name], axis=1)
                elif task_type == 'binary':
                    threshold = use_thresholds.get(name, 0.5)
                    y_pred_binary_dict[name] = (y_pred_dict[name][:, 0] > threshold).astype(int)
                elif task_type == 'multilabel':
                    thresholds_list = use_thresholds.get(name, 0.5)
                    if isinstance(thresholds_list, (float, int)):
                        y_pred_binary_dict[name] = (y_pred_dict[name] > thresholds_list).astype(int)
                    else:
                        thresh_array = np.array(thresholds_list)
                        y_pred_binary_dict[name] = (y_pred_dict[name] > thresh_array).astype(int)
        
        # ------------------------------------------------------------------
        # Compute metrics per output (tasks and pools)
        # ------------------------------------------------------------------
        task_metrics = {}
        pool_metrics = {}
        # Track classification and regression overall metrics separately
        overall_clf_metrics = {m: [] for m in metrics if m in classification_metrics}
        overall_reg_metrics = {m: [] for m in ['mse', 'mae', 'r2', 'pearson_r']}
        
        for name in all_output_names:
            is_pool = self.model.config.is_pool(name)
            
            # Look up task type
            if is_pool:
                task_type = self.model.config.get_pool_task_type(name)
            else:
                task_idx = list(self.model.task_names).index(name) if self.model.task_names else all_output_names.index(name)
                task_type = self.model.config.get_task_type(task_idx)
            
            y_out_true = y_true_dict[name]
            y_out_pred_probs = y_pred_dict[name]
            
            # Filter out masked values
            valid_mask = np.isfinite(y_out_true) & (y_out_true != mask_value)
            if y_out_true.ndim > 1:
                valid_mask = valid_mask[:, 0] if valid_mask.shape[1] > 0 else np.ones(y_out_true.shape[0], dtype=bool)
            
            y_out_true_valid = y_out_true[valid_mask]
            y_out_pred_probs_valid = y_out_pred_probs[valid_mask]
            
            if len(y_out_true_valid) == 0:
                if task_type == 'regression':
                    results_for_output = {m: None for m in ['mse', 'mae', 'r2', 'pearson_r']}
                else:
                    results_for_output = {m: None for m in metrics if m in classification_metrics}
                if is_pool:
                    pool_metrics[name] = results_for_output
                else:
                    task_metrics[name] = results_for_output
                continue
            
            output_results = {}
            
            # ==============================================================
            # REGRESSION METRICS
            # ==============================================================
            if task_type == 'regression':
                y_true_flat = y_out_true_valid.ravel()
                y_pred_flat = y_out_pred_probs_valid.ravel()
                
                mse = float(mean_squared_error(y_true_flat, y_pred_flat))
                mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
                output_results['mse'] = mse
                output_results['mae'] = mae
                
                # R² and Pearson require at least 2 samples
                if len(y_true_flat) >= 2:
                    r2 = float(r2_score(y_true_flat, y_pred_flat))
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(y_true_flat, y_pred_flat)
                    output_results['r2'] = r2
                    output_results['pearson_r'] = float(r)
                else:
                    output_results['r2'] = None
                    output_results['pearson_r'] = None
                
                if not is_pool:
                    overall_reg_metrics['mse'].append(mse)
                    overall_reg_metrics['mae'].append(mae)
                    if output_results['r2'] is not None:
                        overall_reg_metrics['r2'].append(output_results['r2'])
                        overall_reg_metrics['pearson_r'].append(output_results['pearson_r'])
                
                if is_pool:
                    pool_metrics[name] = output_results
                else:
                    task_metrics[name] = output_results
                continue
            
            # ==============================================================
            # CLASSIFICATION METRICS (binary, multiclass, multilabel)
            # ==============================================================
            
            # For pooled binary labels with 'mean' strategy, ground truth
            # may be fractional — binarize for classification metrics.
            y_out_true_binary = y_out_true_valid
            if is_pool and task_type != 'multiclass':
                y_out_true_binary = (y_out_true_valid > 0).astype(int)
                if task_type == 'binary' and y_out_true_binary.ndim > 1:
                    y_out_true_binary = y_out_true_binary[:, 0]
            elif y_out_true_valid.ndim > 1:
                if task_type == 'binary':
                    y_out_true_binary = y_out_true_valid[:, 0]
                else:
                    y_out_true_binary = y_out_true_valid # Keep 2D for multilabel
            
            # Compute binary metrics if thresholds available
            if use_thresholds is not None and task_type not in ('multiclass', 'regression'):
                y_out_pred_binary = y_pred_binary_dict[name][valid_mask]
                avg_method = 'weighted'
                
                if 'accuracy' in metrics:
                    acc = accuracy_score(y_out_true_binary, y_out_pred_binary)
                    output_results['accuracy'] = float(acc)
                    if not is_pool:
                        overall_clf_metrics['accuracy'].append(acc)
                
                if 'precision' in metrics:
                    prec = precision_score(y_out_true_binary, y_out_pred_binary, average=avg_method, zero_division=0)
                    output_results['precision'] = float(prec)
                    if not is_pool:
                        overall_clf_metrics['precision'].append(prec)
                
                if 'recall' in metrics:
                    rec = recall_score(y_out_true_binary, y_out_pred_binary, average=avg_method, zero_division=0)
                    output_results['recall'] = float(rec)
                    if not is_pool:
                        overall_clf_metrics['recall'].append(rec)
                
                if 'f1' in metrics:
                    f1 = f1_score(y_out_true_binary, y_out_pred_binary, average=avg_method, zero_division=0)
                    output_results['f1'] = float(f1)
                    if not is_pool:
                        overall_clf_metrics['f1'].append(f1)
            elif task_type == 'multiclass' and use_thresholds is not None:
                y_out_pred_classes = y_pred_binary_dict[name][valid_mask]
                y_out_true_classes = y_out_true_valid[:, 0] if y_out_true_valid.ndim > 1 else y_out_true_valid
                
                if 'accuracy' in metrics:
                    acc = accuracy_score(y_out_true_classes, y_out_pred_classes)
                    output_results['accuracy'] = float(acc)
                    if not is_pool:
                        overall_clf_metrics['accuracy'].append(acc)
                
                if 'f1' in metrics:
                    f1 = f1_score(y_out_true_classes, y_out_pred_classes, average='weighted', zero_division=0)
                    output_results['f1'] = float(f1)
                    if not is_pool:
                        overall_clf_metrics['f1'].append(f1)
            
            # Compute probability-based metrics
            if 'auc' in metrics and task_type in ['binary', 'multilabel']:
                try:
                    y_probs_for_auc = y_out_pred_probs_valid[:, 0] if task_type == 'binary' and y_out_pred_probs_valid.ndim > 1 else y_out_pred_probs_valid
                    avg_auc = 'macro' if task_type == 'multilabel' else 'macro' # sklearn requires macro/weighted for multilabel ROC
                    auc = roc_auc_score(y_out_true_binary, y_probs_for_auc, average=avg_auc)
                    output_results['auc'] = float(auc)
                    if not is_pool:
                        overall_clf_metrics['auc'].append(auc)
                except ValueError:
                    output_results['auc'] = None
            
            if 'average_precision' in metrics and task_type in ['binary', 'multilabel']:
                try:
                    y_probs_for_ap = y_out_pred_probs_valid[:, 0] if task_type == 'binary' and y_out_pred_probs_valid.ndim > 1 else y_out_pred_probs_valid
                    avg_ap = 'macro' if task_type == 'multilabel' else 'macro'
                    ap = average_precision_score(y_out_true_binary, y_probs_for_ap, average=avg_ap)
                    output_results['average_precision'] = float(ap)
                    if not is_pool:
                        overall_clf_metrics['average_precision'].append(ap)
                except ValueError:
                    output_results['average_precision'] = None
            
            # Store in the right bucket
            if is_pool:
                pool_metrics[name] = output_results
            else:
                task_metrics[name] = output_results
        
        # Aggregate overall metrics: split by task type
        overall_metrics = {}
        
        # Classification averages
        for metric, scores in overall_clf_metrics.items():
            valid_scores = [s for s in scores if s is not None]
            overall_metrics[metric] = float(np.mean(valid_scores)) if valid_scores else None
        
        # Regression averages
        for metric, scores in overall_reg_metrics.items():
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                overall_metrics[metric] = float(np.mean(valid_scores))
        
        # Build results dictionary
        results = {
            'probabilities_dict': y_pred_dict,
            'task_metrics': task_metrics,
            'overall_metrics': overall_metrics
        }
        
        if pool_metrics:
            results['pool_metrics'] = pool_metrics
        
        if y_pred_binary_dict is not None:
            results['predictions_dict'] = y_pred_binary_dict
            results['thresholds_used'] = use_thresholds
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        # Classification metrics
        clf_keys = [k for k in overall_metrics if k in classification_metrics]
        if clf_keys:
            print("Classification (averaged over classification tasks):")
            for metric in clf_keys:
                value = overall_metrics[metric]
                if value is not None:
                    print(f"  {metric.upper():20s}: {value:.4f}")
                else:
                    print(f"  {metric.upper():20s}: N/A")
        
        # Regression metrics
        reg_keys = [k for k in overall_metrics if k in regression_metrics]
        if reg_keys:
            print("Regression (averaged over regression tasks):")
            for metric in reg_keys:
                value = overall_metrics[metric]
                if value is not None:
                    print(f"  {metric.upper():20s}: {value:.4f}")
                else:
                    print(f"  {metric.upper():20s}: N/A")
        
        if pool_metrics:
            print("-" * 60)
            print("Pool Metrics:")
            for pool_name, pm in sorted(pool_metrics.items()):
                metric_strs = []
                for m, v in sorted(pm.items()):
                    metric_strs.append(f"{m}={v:.4f}" if v is not None else f"{m}=N/A")
                print(f"  {pool_name}: {', '.join(metric_strs)}")
        
        print("=" * 60 + "\n")
        
        return results
    
    def save_checkpoint(
        self,
        filepath: str,
        thresholds: Optional[Dict[str, Union[float, List[float], None]]] = None,
        save_format: str = 'tf',
        save_optimizer: bool = True
    ) -> None:
        """Save training checkpoint including model, thresholds, and optimizer state.
        
        Saves everything needed to resume training. Model and thresholds are also
        saved via model.save(), so this checkpoint is self-contained.
        
        Args:
            filepath: Directory path to save checkpoint
            thresholds: Optional dict of task_name -> threshold values. 
                        If None, uses self.optimal_thresholds, then 
                        self.thresholds if available. optimal_thresholds is 
                        populated from fit() while thresholds is set in load_checkpoint
                        (i.e. passed explicitly).
            save_format: Format to save weights ('tf' or 'h5')
            save_optimizer: Whether to save optimizer state for training resumption.
                          Use False for deployment-only to save space (~2x reduction).
            
        Example:
            >>> history, thresholds = trainer.fit(train_ds, val_ds, optimize_thresholds=True)
            >>> trainer.save_checkpoint('checkpoints/model_v1', thresholds)
            >>> 
            >>> # Load for inference
            >>> trainer, thresholds = Trainer.load_checkpoint('checkpoints/model_v1')
            >>> 
            >>> # Load for continued training (restores optimizer)
            >>> trainer, thresholds = Trainer.load_checkpoint(
            ...     'checkpoints/model_v1', load_optimizer=True, compile_model=True)
        """
        # Use explicitly provided thresholds first; otherwise prefer thresholds
        # discovered during fit(), with legacy fallback for checkpoint-loaded trainers.
        thresholds_to_save = thresholds
        if thresholds_to_save is None:
            thresholds_to_save = getattr(self, 'optimal_thresholds', None)
        if thresholds_to_save is None:
            thresholds_to_save = getattr(self, 'thresholds', None)
        
        # Save model with thresholds (includes weights, config, encoder, thresholds)
        self.model.save(filepath, save_format=save_format, thresholds=thresholds_to_save)
        
        # Save optimizer state if requested and model is compiled
        if save_optimizer and hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_weights_path = os.path.join(filepath, 'optimizer.pkl')
            try:
                import pickle
                # Get optimizer weights (includes momentum, running averages, etc.)
                optimizer_weights = self.model.optimizer.get_weights()
                with open(optimizer_weights_path, 'wb') as f:
                    pickle.dump(optimizer_weights, f)
                print(f"  - Optimizer state: {optimizer_weights_path}")
            except Exception as e:
                print(f"    Warning: Could not save optimizer state: {e}")
        
        # Save training config (for reference, not needed for loading)
        training_config_dict = {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay_rate': self.config.weight_decay_rate,
            'warmup_percent': self.config.warmup_percent,
            'optimizer_type': self.config.optimizer_type,
            'dropout_rate': self.config.dropout_rate,
            'early_stopping_patience': self.config.early_stopping_patience,
            'lr_schedule_type': self.config.lr_schedule_type,
        }
        
        with open(os.path.join(filepath, 'training_config.json'), 'w') as f:
            json.dump(training_config_dict, f, indent=2)
        
        print(f"Checkpoint saved to {filepath}/")
    
    @staticmethod
    def load_checkpoint(
        filepath: str,
        encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
        custom_objects: Optional[dict] = None,
        load_optimizer: bool = True,
        compile_model: bool = False
    ) -> Tuple['Trainer', Optional[Dict[str, Union[float, List[float], None]]]]:
        """Load a complete training checkpoint.
        
        Restores a Trainer instance with model and training state from a saved checkpoint.
        
        Args:
            filepath: Directory path containing saved checkpoint
            encoder: Optional encoder override (see load_model documentation)
            custom_objects: Optional dict of custom objects for deserialization
            load_optimizer: Whether to restore optimizer state for training resumption.
                          Only works if model is compiled first.
            compile_model: Whether to compile the model after loading. Required if
                         load_optimizer=True. Uses saved training config.
            
        Returns:
            Tuple of (Trainer instance, thresholds dict or None)
            
        Example:
            >>> # Load for inference only
            >>> trainer, thresholds = Trainer.load_checkpoint('checkpoints/best_model')
            >>> results = trainer.evaluate(test_dataset, thresholds=thresholds)
            >>> 
            >>> # Load for continued training (restores optimizer)
            >>> trainer, thresholds = Trainer.load_checkpoint(
            ...     'checkpoints/best_model',
            ...     load_optimizer=True,
            ...     compile_model=True
            ... )
            >>> # Continue training with restored optimizer state
            >>> history, new_thresholds = trainer.fit(train_ds, val_ds, num_tasks=10)
        """
        from ..models.persistence import load_model
        
        # Load model (includes weights, config, encoder, AND thresholds if saved)
        model, thresholds = load_model(filepath, encoder=encoder)
        
        # Load training config if available
        training_config_path = os.path.join(filepath, 'training_config.json')
        if os.path.exists(training_config_path):
            with open(training_config_path, 'r') as f:
                training_config_dict = json.load(f)
            config = TrainingConfig(**training_config_dict)
        else:
            # Use default config
            config = TrainingConfig()
        
        # Create trainer
        trainer = Trainer(model=model, config=config)
        
        # Attach thresholds if loaded
        if thresholds is not None:
            trainer.thresholds = thresholds
        
        # Compile model if requested (needed for optimizer loading)
        if compile_model:
            # Build a basic loss function for compilation
            # User will need to call fit() which rebuilds with proper weights
            def dummy_loss(y_true, y_pred):
                return tf.reduce_mean(y_pred)  # Placeholder
            
            model.compile(
                optimizer=trainer._build_optimizer(steps_per_epoch=100),
                loss=dummy_loss
            )
        
        # Load optimizer state if available and requested
        if load_optimizer and compile_model:
            optimizer_weights_path = os.path.join(filepath, 'optimizer.pkl')
            if os.path.exists(optimizer_weights_path):
                try:
                    import pickle
                    with open(optimizer_weights_path, 'rb') as f:
                        optimizer_weights = pickle.load(f)
                    
                    # Need to call the model once to initialize optimizer variables
                    # This is a TensorFlow requirement before setting optimizer weights
                    dummy_input = tf.zeros((1, 128))  # Placeholder
                    if model.uses_encoder:
                        if model.encoder_type == 'huggingface':
                            dummy_input = {
                                'input_ids': tf.zeros((1, 128), dtype=tf.int32),
                                'attention_mask': tf.ones((1, 128), dtype=tf.int32),
                                'token_type_ids': tf.zeros((1, 128), dtype=tf.int32),
                            }
                        else:
                            dummy_input = {
                                'input_word_ids': tf.zeros((1, 128), dtype=tf.int32),
                                'input_mask': tf.ones((1, 128), dtype=tf.int32),
                                'input_type_ids': tf.zeros((1, 128), dtype=tf.int32),
                            }
                    
                    with tf.GradientTape() as tape:
                        _ = model(dummy_input, training=True)
                    
                    # Now set optimizer weights
                    model.optimizer.set_weights(optimizer_weights)
                    print(f"  - Optimizer state restored")
                except Exception as e:
                    print(f"    Warning: Could not restore optimizer state: {e}")
            elif load_optimizer:
                print(f"    No optimizer state found at {optimizer_weights_path}")
        
        print(f"Checkpoint loaded from {filepath}/")
        
        return trainer, thresholds

