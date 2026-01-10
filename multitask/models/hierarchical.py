"""
Multi-task learning model with hierarchical branching.

This module implements a flexible hierarchical branching architecture for multi-task learning.
"""

import copy
import tensorflow as tf
from typing import Optional, Union, Dict, List, Any, Tuple
from multitask.config import ModelConfig


class MultiTaskModel(tf.keras.Model):
    """Multi-task learning model with flexible hierarchical branching.
    
    This model supports arbitrary hierarchical task structures where each branch
    can have a different number of children. The architecture consists of:
    
    1. Optional encoder (TensorFlow Hub, HuggingFace, or custom) or pre-computed embeddings
    2. Optional shared layers (common to all tasks)
    3. Hierarchical branching layers (configurable depth and branching factor)
    4. Output layers (one per final task)
    
    Args:
        config: ModelConfig object defining the architecture
        encoder: Optional encoder - TensorFlow Hub URL (https://tfhub.dev/...),
                 HuggingFace model name (e.g., 'bert-base-uncased'), or custom Keras layer
        trainable_encoder: Whether encoder weights are trainable
        embedding_dim: If using embeddings, their dimension
        num_classes: Number of classes per task (default: 2 for binary)
        dropout: Dropout rate (overrides config if provided)
        name: Optional model name
    
    Example:
        >>> # Using TensorFlow Hub
        >>> config = ModelConfig(task_structure=[[2]])
        >>> model = MultiTaskModel(config=config, encoder='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
        >>> 
        >>> # Using HuggingFace
        >>> model = MultiTaskModel(config=config, encoder='bert-base-uncased')
    """
    
    def __init__(
        self,
        config: ModelConfig,
        encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
        trainable_encoder: bool = True,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
        dropout: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        # Use dropout from config if not explicitly provided
        dropout = dropout if dropout is not None else 0.1
        
        super().__init__(name=name or "MultiTaskModel", **kwargs)
        
        self.config = config
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.task_names = config.task_names
        
        # Validate configuration
        self._validate_config()
        
        # Build encoder
        self._build_encoder(encoder, trainable_encoder, embedding_dim)
        
        # Build architecture
        self._build_shared_layers()
        self._build_branch_layers()
        self._build_output_layers()
    
    def _validate_config(self):
        """Validate model configuration."""
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout must be in range [0, 1)")
    
    def _build_encoder(
        self,
        encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
        trainable_encoder: bool = True,
        embedding_dim: Optional[int] = None
    ):
        """Build the encoder component.
        
        Args:
            encoder: Either a TensorFlow Hub URL, HuggingFace model name, a Keras layer, or None for embedding input
            trainable_encoder: Whether encoder weights should be trainable
            embedding_dim: If using pre-computed embeddings, the dimension size
        """
        if encoder is not None:
            # Using a pre-trained encoder
            if isinstance(encoder, str):
                # Store identifier for save/load
                self._encoder_identifier = encoder
                
                if encoder.startswith('https://tfhub.dev/'):
                    # TensorFlow Hub model
                    import tensorflow_hub as hub
                    self.encoder = hub.KerasLayer(encoder, trainable=trainable_encoder, name='encoder')
                    self.encoder_type = 'tfhub'
                else:
                    # HuggingFace model (e.g., 'bert-base-uncased', 'roberta-large')
                    try:
                        from transformers import TFAutoModel
                    except ImportError:
                        raise ImportError(
                            "HuggingFace transformers library is required for this encoder. "
                            "Install with: pip install transformers>=4.30.0 or pip install multitask[huggingface]"
                        )
                    self.encoder = TFAutoModel.from_pretrained(encoder, name='encoder')
                    self.encoder.trainable = trainable_encoder
                    self.encoder_type = 'huggingface'
            else:
                # Custom Keras layer
                self.encoder = encoder
                self.encoder_type = 'custom'
                self._encoder_identifier = None
            self.uses_encoder = True
            self.embedding_dim = None
        else:
            # Using pre-computed embeddings
            if embedding_dim is None:
                raise ValueError("Must specify embedding_dim when not using an encoder")
            self.encoder = None
            self.encoder_type = None
            self._encoder_identifier = None
            self.uses_encoder = False
            self.embedding_dim = embedding_dim
    
    def _encode_input(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Encode input through encoder or pass through embeddings.
        
        Args:
            inputs: Raw inputs (text tokens) or pre-computed embeddings
            training: Whether in training mode
            
        Returns:
            Encoded representation
        """
        if self.uses_encoder:
            encoder_output = self.encoder(inputs, training=training)
            
            # Handle different encoder output formats
            if self.encoder_type == 'huggingface':
                # HuggingFace models return objects with attributes
                if hasattr(encoder_output, 'pooler_output') and encoder_output.pooler_output is not None:
                    return encoder_output.pooler_output
                elif hasattr(encoder_output, 'last_hidden_state'):
                    # Use [CLS] token (first token) if no pooler
                    return encoder_output.last_hidden_state[:, 0, :]
                else:
                    # Fallback to first element if it's a tuple/list
                    return encoder_output[0] if isinstance(encoder_output, (tuple, list)) else encoder_output
            elif isinstance(encoder_output, dict):
                # TF Hub dict format
                return encoder_output.get('pooled_output', encoder_output.get('default', encoder_output))
            else:
                # Direct tensor output
                return encoder_output
        else:
            # Already embeddings, just pass through
            return inputs
    
    def _build_shared_layers(self):
        """Build optional shared layers before branching."""
        self.shared_layers = []
        
        if self.config.shared_layer_sizes is not None:
            for i, size in enumerate(self.config.shared_layer_sizes):
                self.shared_layers.append(
                    tf.keras.layers.Dense(size, activation='relu', name=f'shared_dense_{i}')
                )
                self.shared_layers.append(
                    tf.keras.layers.Dropout(self.dropout_rate, name=f'shared_dropout_{i}')
                )
    
    def _build_branch_layers(self):
        """Build hierarchical branching layers based on task_structure."""
        self.branch_layers = []
        
        # Build layers for each branching level
        for level_idx in range(self.config.num_branch_levels):
            level_branches = self.config.get_branches_at_level(level_idx)
            layer_sizes = self.config.get_layer_sizes(level_idx)
            
            # Create layers for each branch at this level
            branch_level_layers = []
            for branch_idx in range(level_branches):
                branch_layers = []
                for layer_idx, size in enumerate(layer_sizes):
                    branch_layers.append(
                        tf.keras.layers.Dense(
                            size, 
                            activation='relu',
                            name=f'branch_L{level_idx}_B{branch_idx}_dense_{layer_idx}'
                        )
                    )
                    branch_layers.append(
                        tf.keras.layers.Dropout(
                            self.dropout_rate,
                            name=f'branch_L{level_idx}_B{branch_idx}_dropout_{layer_idx}'
                        )
                    )
                branch_level_layers.append(branch_layers)
            
            self.branch_layers.append(branch_level_layers)
    
    def _build_output_layers(self):
        """Build output layers for final predictions.
        
        Each task gets one or more output units depending on its type:
        - Binary: 1 unit with sigmoid activation
        - Multiclass: K units with softmax activation
        - Multilabel: K units with sigmoid activation
        """
        self.output_layers = {}
        
        for task_idx in range(self.config.total_outputs):
            task_type = self.config.get_task_type(task_idx)
            num_classes = self.config.get_num_classes(task_idx)
            output_dim = self.config.get_output_dim_for_task(task_idx)
            task_name = self.task_names[task_idx] if self.task_names else f'task_{task_idx}'
            
            if task_type == 'binary':
                # Binary classification: 1 output with sigmoid
                layer = tf.keras.layers.Dense(
                    output_dim,
                    activation='sigmoid',
                    name=f'{task_name}_output'
                )
            elif task_type == 'multiclass':
                # Multiclass: K outputs with softmax
                layer = tf.keras.layers.Dense(
                    output_dim,
                    activation='softmax',
                    name=f'{task_name}_output'
                )
            elif task_type == 'multilabel':
                # Multilabel: K outputs with sigmoid
                layer = tf.keras.layers.Dense(
                    output_dim,
                    activation='sigmoid',
                    name=f'{task_name}_output'
                )
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            self.output_layers[task_name] = layer
    
    
    def call(
        self, 
        inputs: Union[tf.Tensor, Dict[str, tf.Tensor]], 
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass through the hierarchical model.
        
        Args:
            inputs: Either preprocessed text tokens (if using encoder) or 
                    pre-computed embeddings
            training: Whether in training mode
            
        Returns:
            Concatenated predictions for all tasks, shape [batch_size, num_outputs]
        """
        # Encode input
        x = self._encode_input(inputs, training=training)
        
        # Apply shared layers
        for layer in self.shared_layers:
            x = layer(x, training=training)
        
        # Store representation from shared layers
        shared_rep = x
        
        # Process hierarchical branches
        branch_reps = [shared_rep]  # Start with shared representation
        
        for level_idx, level_layers in enumerate(self.branch_layers):
            next_level_reps = []
            
            # Determine which parent representations to use
            if level_idx == 0:
                # First level: all branches use shared representation
                num_parent_branches = 1
                parent_start_idx = 0
            else:
                # Subsequent levels: each branch uses its parent's representation
                num_parent_branches = self.config.get_branches_at_level(level_idx - 1)
                parent_start_idx = 0
            
            # Process each branch at this level
            branch_idx_in_level = 0
            for parent_idx in range(num_parent_branches):
                parent_rep = branch_reps[parent_start_idx + parent_idx] if level_idx > 0 else shared_rep
                num_children = self.config.task_structure[level_idx][parent_idx]
                
                # Create representation for each child branch
                for child_idx in range(num_children):
                    x = parent_rep
                    for layer in level_layers[branch_idx_in_level]:
                        x = layer(x, training=training)
                    next_level_reps.append(x)
                    branch_idx_in_level += 1
            
            branch_reps = next_level_reps
        
        # Apply output layers to final branch representations
        # Return as dictionary mapping task names to outputs
        output_dict = {}
        for task_idx, branch_rep in enumerate(branch_reps):
            task_name = self.task_names[task_idx] if self.task_names else f'task_{task_idx}'
            output = self.output_layers[task_name](branch_rep, training=training)
            output_dict[task_name] = output
        
        return output_dict
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'config': {
                'task_structure': self.config.task_structure,
                'shared_layer_sizes': self.config.shared_layer_sizes,
                'branch_layer_sizes': self.config.branch_layer_sizes,
                'default_layer_sizes': self.config.default_layer_sizes,
                'trainable_encoder': self.config.trainable_encoder,
                'pooling_strategy': self.config.pooling_strategy,
            },
            'uses_encoder': self.uses_encoder,
            'embedding_dim': self.embedding_dim,
        })
        return config
    
    def summary_tree(self) -> str:
        """Generate a human-readable summary of the branching structure.
        
        Returns:
            String representation of the hierarchical structure
        """
        lines = ["Hierarchical Multi-Task Model Structure:"]
        lines.append("=" * 50)
        
        if self.uses_encoder:
            lines.append("Encoder: TensorFlow Hub Layer")
        else:
            lines.append(f"Input: Pre-computed Embeddings (dim={self.embedding_dim})")
        
        if self.shared_layers:
            lines.append(f"\nShared Layers: {len(self.shared_layers) // 2} dense layers")
        
        lines.append("\nHierarchical Branching:")
        for level_idx, level_structure in enumerate(self.config.task_structure):
            indent = "  " * level_idx
            lines.append(f"{indent}Level {level_idx}: {level_structure}")
        
        lines.append(f"\nTotal Outputs: {self.config.total_outputs}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def get_task_info(self) -> List[Dict[str, Any]]:
        """Get information about each task output.
        
        Returns a list where each element corresponds to a model output and contains:
        - 'output_idx': The index of this output in the model's output tensor
        - 'name': Task name if task_names provided, otherwise 'task_{idx}'
        - 'parent_branch_path': List of branch indices tracing from root to this output
        
        The parent_branch_path shows the hierarchical path through the branching structure.
        For example, [1, 2] means this output came from the 3rd child (index 2) of the 
        2nd parent branch (index 1).
        
        Returns:
            List of dictionaries with task information
            
        Example:
            >>> config = ModelConfig(
            ...     task_structure=[[3], [2, 5, 3]],
            ...     task_names=['P1', 'P2', 'I1', 'I2', 'I3', 'I4', 'I5', 'W1', 'W2', 'W3']
            ... )
            >>> model = MultiTaskModel(config)
            >>> info = model.get_task_info()
            >>> info[0]
            {'output_idx': 0, 'name': 'P1', 'parent_branch_path': [0, 0]}
            >>> info[7]
            {'output_idx': 7, 'name': 'W1', 'parent_branch_path': [2, 0]}
        """
        task_info = []
        output_idx = 0
        
        # Get the final level of the hierarchy (the one that produces outputs)
        final_level = self.config.task_structure[-1]
        
        # Trace back through the hierarchy to find parent paths
        # We need to work through the structure to map each output to its parent path
        
        if len(self.config.task_structure) == 1:
            # Simple case: only one level, each output is a direct branch
            for branch_idx in range(final_level[0]):
                task_info.append({
                    'output_idx': output_idx,
                    'name': self.task_names[output_idx] if self.task_names else f'task_{output_idx}',
                    'parent_branch_path': [branch_idx]
                })
                output_idx += 1
        else:
            # Multi-level hierarchy: trace through the structure
            # Build paths by traversing the hierarchy level by level
            parent_paths = self._build_parent_paths()
            
            for path in parent_paths:
                task_info.append({
                    'output_idx': output_idx,
                    'name': self.task_names[output_idx] if self.task_names else f'task_{output_idx}',
                    'parent_branch_path': path
                })
                output_idx += 1
        
        return task_info
    
    def _build_parent_paths(self) -> List[List[int]]:
        """Build the hierarchical paths for all outputs.
        
        Returns:
            List of paths, where each path is a list of branch indices from root to leaf
        """
        paths = []
        
        # Start with paths at level 0 (initial branches)
        num_initial_branches = self.config.task_structure[0][0]
        current_paths = [[i] for i in range(num_initial_branches)]
        
        # Traverse through each level
        for level_idx in range(1, len(self.config.task_structure)):
            next_paths = []
            level_structure = self.config.task_structure[level_idx]
            
            # For each parent path, add children according to level_structure
            structure_idx = 0
            for parent_path in current_paths:
                num_children = level_structure[structure_idx]
                
                # Create a path for each child
                for child_idx in range(num_children):
                    child_path = parent_path + [child_idx]
                    next_paths.append(child_path)
                
                structure_idx += 1
            
            current_paths = next_paths
        
        return current_paths    
    def flatten_outputs(
        self, 
        output_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, Tuple[int, int]]]:
        """Convert dictionary outputs to concatenated tensor for external tools.
        
        Useful when integrating with external tools that expect tensor outputs.
        Provides mapping of where each task's output is located in the concatenated tensor.
        
        Args:
            output_dict: Dictionary output from model.predict() or model()
            
        Returns:
            Tuple of:
                - concatenated_tensor: [batch_size, total_output_dims]
                - task_indices: Dict mapping task_name -> (start_col, end_col) in tensor
                
        Example:
            >>> predictions = model.predict(data)  # Returns dict
            >>> tensor, indices = model.flatten_outputs(predictions)
            >>> # tensor shape might be [32, 14] for mixed binary/multiclass
            >>> # indices = {'task_0': (0, 1), 'task_1': (1, 2), 'task_2': (2, 7), ...}
        """
        task_names = sorted(output_dict.keys())
        tensors = [output_dict[name] for name in task_names]
        concatenated = tf.concat(tensors, axis=1)
        
        # Track where each task's output is
        task_indices = {}
        start = 0
        for task_name in task_names:
            num_dims = output_dict[task_name].shape[-1]
            task_indices[task_name] = (start, start + num_dims)
            start += num_dims
        
        return concatenated, task_indices
    
    def unflatten_outputs(
        self, 
        concatenated_tensor: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Convert concatenated tensor back to dictionary format.
        
        Reverse of flatten_outputs. Useful after processing with external tools.
        
        Args:
            concatenated_tensor: Tensor output shape [batch_size, total_output_dims]
            
        Returns:
            Dictionary with task outputs matching original structure
            
        Example:
            >>> external_result = external_tool(tensor_output)
            >>> output_dict = model.unflatten_outputs(external_result)
            >>> # output_dict['task_0'] has shape [batch_size, 1]
            >>> # output_dict['task_2'] has shape [batch_size, 5]
        """
        if self.task_names is None:
            task_names = [f'task_{i}' for i in range(self.config.total_outputs)]
        else:
            task_names = self.task_names
        
        output_dict = {}
        start = 0
        for task_name in task_names:
            output_dim = self.config.get_output_dim_for_task(task_names.index(task_name))
            output_dict[task_name] = concatenated_tensor[:, start:start+output_dim]
            start += output_dim
        
        return output_dict
    
    def validate_dataset(
        self,
        dataset: tf.data.Dataset,
        num_batches: int = 3,
        verbose: bool = True
    ) -> bool:
        """Validate dataset format and structure before training.
        
        Checks:
        - Dataset yields (inputs, labels_dict) tuples
        - Input format matches model type (dict for encoder, tensor for embeddings)
        - Input shapes are reasonable
        - Labels dict has correct structure and task names
        - Label shapes and values are valid for their task types
        - Batch consistency across multiple batches
        
        Args:
            dataset: TensorFlow Dataset to validate
            num_batches: Number of batches to sample for validation (default: 3)
            verbose: Whether to print validation results (default: True)
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: With detailed error message if validation fails
        """
        if verbose:
            print("\nValidating dataset structure...")
            print("=" * 60)
        
        batch_count = 0
        last_batch_sizes = []
        
        try:
            for inputs, labels_dict in dataset.take(num_batches):
                batch_count += 1
                # Get batch size from inputs
                if isinstance(inputs, (list, tuple)):
                    batch_size = inputs[0].shape[0]
                elif isinstance(inputs, dict):
                    batch_size = next(iter(inputs.values())).shape[0]
                else:
                    batch_size = inputs.shape[0]
                last_batch_sizes.append(batch_size)
                
                # ===== VALIDATE INPUTS =====
                if self.uses_encoder:
                    # Expecting dict (BERT, etc.) or tensor input
                    if isinstance(inputs, dict):
                        # Dict input (typical for BERT)
                        for key, value in inputs.items():
                            if not tf.is_tensor(value):
                                raise ValueError(
                                    f"Input dict value for key '{key}' is not a tensor. "
                                    f"Type: {type(value)}"
                                )
                            if len(value.shape) < 2:
                                raise ValueError(
                                    f"Input tensor '{key}' has shape {value.shape}. "
                                    f"Expected at least 2D (batch_size, sequence_length)."
                                )
                            if value.shape[0] != batch_size:
                                raise ValueError(
                                    f"Batch size mismatch in input '{key}': "
                                    f"{value.shape[0]} vs {batch_size}"
                                )
                    elif tf.is_tensor(inputs):
                        # Tensor input
                        if len(inputs.shape) < 2:
                            raise ValueError(
                                f"Input tensor has shape {inputs.shape}. "
                                f"Expected at least 2D (batch_size, sequence_length)."
                            )
                    else:
                        raise ValueError(
                            f"Input must be dict or tensor. Got {type(inputs)}"
                        )
                else:
                    # Expecting embedding tensor
                    if not tf.is_tensor(inputs):
                        raise ValueError(
                            f"Model uses embeddings but input is {type(inputs)}. "
                            f"Expected tensor shape [batch_size, {self.embedding_dim}]"
                        )
                    if len(inputs.shape) != 2:
                        raise ValueError(
                            f"Embedding input must be 2D but got shape {inputs.shape}. "
                            f"Expected [batch_size, embedding_dim]"
                        )
                    if inputs.shape[1] != self.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch: got {inputs.shape[1]}, "
                            f"expected {self.embedding_dim}"
                        )
                
                # ===== VALIDATE LABELS =====
                if not isinstance(labels_dict, dict):
                    raise ValueError(
                        f"Labels must be a dictionary mapping task names to label tensors. "
                        f"Got {type(labels_dict)}"
                    )
                
                # Check task names
                expected_task_names = set(self.task_names) if self.task_names else set(f'task_{i}' for i in range(self.config.total_outputs))
                actual_task_names = set(labels_dict.keys())
                
                if expected_task_names != actual_task_names:
                    raise ValueError(
                        f"Label task names mismatch.\n"
                        f"Expected: {sorted(expected_task_names)}\n"
                        f"Got: {sorted(actual_task_names)}"
                    )
                
                # Validate each task's labels
                for task_idx, task_name in enumerate(sorted(labels_dict.keys())):
                    labels = labels_dict[task_name]
                    task_type = self.config.get_task_type(task_idx)
                    num_classes = self.config.get_num_classes(task_idx)
                    
                    # Check tensor
                    if not tf.is_tensor(labels):
                        raise ValueError(
                            f"Labels for '{task_name}' must be tensor, got {type(labels)}"
                        )
                    
                    # Check batch size matches
                    if labels.shape[0] != batch_size:
                        raise ValueError(
                            f"Labels batch size mismatch for '{task_name}': "
                            f"{labels.shape[0]} vs {batch_size}"
                        )
                    
                    # Check shape and values based on task type
                    if task_type == 'binary':
                        # Binary: [batch_size] or [batch_size, 1], values in {0, 1, -1}
                        if len(labels.shape) == 2 and labels.shape[1] != 1:
                            raise ValueError(
                                f"Binary task '{task_name}' has shape {labels.shape}. "
                                f"Expected [batch_size] or [batch_size, 1]"
                            )
                        
                        label_values = tf.unique(tf.reshape(labels, [-1]))[0].numpy()
                        invalid_values = [v for v in label_values if v not in [-1, 0, 1]]
                        if invalid_values:
                            raise ValueError(
                                f"Binary task '{task_name}' has invalid values {invalid_values}. "
                                f"Expected 0, 1, or -1 (mask)"
                            )
                    
                    elif task_type == 'multiclass':
                        # Multiclass: [batch_size], values in {0..K-1, -1}
                        if len(labels.shape) != 1:
                            raise ValueError(
                                f"Multiclass task '{task_name}' has shape {labels.shape}. "
                                f"Expected [batch_size] (class indices)"
                            )
                        
                        label_values = tf.unique(labels)[0].numpy()
                        valid_values = set(range(num_classes)) | {-1}
                        invalid_values = [v for v in label_values if v not in valid_values]
                        if invalid_values:
                            raise ValueError(
                                f"Multiclass task '{task_name}' (num_classes={num_classes}) "
                                f"has invalid values {invalid_values}. "
                                f"Expected 0..{num_classes-1} or -1 (mask)"
                            )
                    
                    elif task_type == 'multilabel':
                        # Multilabel: [batch_size, num_classes], values in {0, 1, -1}
                        if len(labels.shape) != 2 or labels.shape[1] != num_classes:
                            raise ValueError(
                                f"Multilabel task '{task_name}' has shape {labels.shape}. "
                                f"Expected [batch_size, {num_classes}]"
                            )
                        
                        label_values = tf.unique(tf.reshape(labels, [-1]))[0].numpy()
                        invalid_values = [v for v in label_values if v not in [-1, 0, 1]]
                        if invalid_values:
                            raise ValueError(
                                f"Multilabel task '{task_name}' has invalid values {invalid_values}. "
                                f"Expected 0, 1, or -1 (mask)"
                            )
                    
                    # Warn if all labels are masked
                    num_valid = tf.reduce_sum(tf.cast(labels != -1, tf.int32)).numpy()
                    if num_valid == 0:
                        if verbose:
                            print(f"Warning: All labels are masked (-1) for task '{task_name}'")
                
                if verbose:
                    print(f"Batch {batch_count}: inputs OK, {len(labels_dict)} task labels OK")
        
        except StopIteration:
            if batch_count == 0:
                raise ValueError("Dataset is empty!")
        except Exception as e:
            if verbose:
                print(f"\nValidation failed:")
                print(f"   {str(e)}")
                print("=" * 60 + "\n")
            raise
        
        # Check batch consistency
        if len(last_batch_sizes) > 1 and len(set(last_batch_sizes[:-1])) > 1:
            if verbose:
                print(f"\nWarning: Batch sizes vary: {last_batch_sizes}")
        
        if verbose:
            print("=" * 60)
            print(f"Dataset validation passed! ({batch_count} batches sampled)")
            print("=" * 60 + "\n")
        
        return True
    
    def save(self, filepath: str, save_format: str = 'tf', thresholds: Optional[Dict[str, Optional[float]]] = None) -> None:
        """Save model with all necessary components for deployment.
        
        Saves:
        - Model weights (architecture + parameters)
        - Model config (task structure, types, num classes)
        - Encoder information (type, URL/model name)
        - Optimized thresholds (if provided)
        - Model metadata
        
        Args:
            filepath: Directory path to save model artifacts
            save_format: Format to save weights ('tf' or 'h5')
            thresholds: Optional dict of task_name -> threshold values for classification.
                       Essential for binary/multilabel tasks. If None, thresholds must be
                       provided during inference via evaluate() or predict().
            
        Example:
            >>> # Save with optimized thresholds
            >>> history, thresholds = trainer.fit(train_ds, val_ds, optimize_thresholds=True)
            >>> model.save('my_model', thresholds=thresholds)
            >>> 
            >>> # Load later
            >>> from multitask import load_model
            >>> model = load_model('my_model')  # Thresholds included
        """
        import json
        import os
        
        # Create save directory
        os.makedirs(filepath, exist_ok=True)
        
        # Save model weights
        # Note: Keras requires .weights.h5 extension for save_weights()
        weights_path = os.path.join(filepath, 'weights.weights.h5')
        self.save_weights(weights_path)
        
        # Save ModelConfig
        config_dict = {
            'task_structure': self.config.task_structure,
            'task_types': self.config.task_types,
            'num_classes_per_task': self.config.num_classes_per_task,
            'task_names': self.config.task_names,
            'shared_layer_sizes': self.config.shared_layer_sizes,
            'branch_layer_sizes': self.config.branch_layer_sizes,
            'default_layer_sizes': self.config.default_layer_sizes,
            'trainable_encoder': self.config.trainable_encoder,
            'pooling_strategy': self.config.pooling_strategy,
        }
        
        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save metadata (encoder info, model settings)
        metadata = {
            'encoder_type': self.encoder_type,
            'uses_encoder': self.uses_encoder,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        }
        
        # Save encoder identifier if using encoder
        if self.uses_encoder and hasattr(self, '_encoder_identifier'):
            metadata['encoder_identifier'] = self._encoder_identifier
        
        with open(os.path.join(filepath, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save thresholds if provided
        if thresholds is not None:
            with open(os.path.join(filepath, 'thresholds.json'), 'w') as f:
                json.dump(thresholds, f, indent=2)
        
        print(f"Model saved to {filepath}/")
        print(f"  - Weights: {weights_path}")
        print(f"  - Config: {os.path.join(filepath, 'config.json')}")
        print(f"  - Metadata: {os.path.join(filepath, 'metadata.json')}")
        if thresholds is not None:
            print(f"  - Thresholds: {os.path.join(filepath, 'thresholds.json')}")
        print(f"  - Weights: {weights_path}")
        print(f"  - Config: {os.path.join(filepath, 'config.json')}")
        print(f"  - Metadata: {os.path.join(filepath, 'metadata.json')}")
