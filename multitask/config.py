"""
Configuration and default hyperparameters

This module defines the default configuration for models, training, and
evaluation. Users can override these defaults by passing arguments to
the relevant functions.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Default configuration for training multi-task models."""
    
    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay_rate: float = 0.01
    warmup_percent: int = 5
    
    # Model hyperparameters
    dropout: float = 0.1
    
    # Sequence/data parameters
    max_sequence_length: int = 512
    
    # Distributed training
    num_workers: int = 1
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # Optimizer settings
    # 'adamw' ran into errors for me locally
    optimizer_type: str = 'adam'  # 'adamw', 'adam', 'sgd'
    
    # Learning rate schedule settings
    use_lr_schedule: bool = True
    lr_schedule_type: str = 'polynomial_decay'  # 'polynomial_decay', 'exponential_decay', 'cosine_decay', 'constant'
    # Also some errors when using warmup
    use_warmup: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in range [0, 1)")
        if self.max_sequence_length > 512:
            raise ValueError("max_sequence_length cannot exceed 512 for BERT-based models")


@dataclass
class ModelConfig:
    """Default configuration for multi-task model architecture.
    
    This configuration supports arbitrary hierarchical branching structures
    where each branch can have a different number of children.
    
    Task Structure Format:
        A list of lists, where each inner list specifies the number of children
        for each branch at that level.
        
        - Level 0: List with one element (number of initial branches from encoder)
        - Level N: List with one element per branch from level N-1
        
    Example branching structures:
        - [[2]]: 2 branches (minimum structure, 2 outputs)
        
        - [[3], [2, 5, 3]]: 
            * Level 0: 3 branches
            * Level 1: branch 0 - 2 children, branch 1 - 5 children, branch 2 - 3 children
            * Total outputs: 2 + 5 + 3 = 10
        
        - [[2], [3, 4], [1, 2, 1, 1, 1, 2]]:
            * Level 0: 2 branches
            * Level 1: branch 0 - 3 children, branch 1 - 4 children (7 total branches)
            * Level 2: each of 7 branches splits as [1, 2, 1, 1, 1, 2]
            * Outputs on each of 7 branches: 1 + 2 + 1 + 1 + 1 + 2 = 8
    
    Layer sizes can be specified per branch level, or use defaults for all levels.
    """
    
    # Task structure: nested list defining branching hierarchy
    # Format: [[n_initial], [n_children_0, n_children_1, ...], [...], ...]
    task_structure: List[List[int]] = field(default_factory=lambda: [[3], [2, 5, 3]])
    
    # Layer sizes for each branch level
    # If None, defaults are used. Can specify different sizes per level.
    # Format: List of layer size lists, one per branch level
    # e.g., [[256, 128], [128, 64], [64, 32]] for a 3-level hierarchy
    branch_layer_sizes: Optional[List[List[int]]] = None
    
    # Default layer sizes if branch_layer_sizes is not specified
    default_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    
    # Optional shared encoder layers before branching
    shared_layer_sizes: Optional[List[int]] = field(default_factory=lambda: [256, 128])
    
    # Encoder settings
    trainable_encoder: bool = True
    
    # Pooling strategy for hierarchical outputs (if applicable)
    # Options: 'concat', 'max', 'mean', 'sum', None (no pooling)
    pooling_strategy: Optional[str] = None
    
    # Optional task names for model outputs
    # If provided, must be a flat list with length equal to total_outputs
    # These correspond to the final level outputs in order
    task_names: Optional[List[str]] = None
    
    # Task types: 'binary', 'multiclass', or 'multilabel'
    # If provided, must be a flat list with length equal to total_tasks (final level outputs)
    # Defaults to 'binary' for all tasks if not provided
    task_types: Optional[List[str]] = None
    
    # Number of classes per task (for multiclass tasks)
    # Only needed for 'multiclass' tasks. Binary tasks default to 2, multiclass needs explicit count.
    # Must be a list of length equal to total_tasks
    # Example: [2, 2, 5, 2, 2, 3, 2, 2, 2, 2] for 10 tasks (some binary, some multiclass)
    num_classes_per_task: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate task structure
        if not self.task_structure or len(self.task_structure) == 0:
            raise ValueError("task_structure must contain at least one branching level")
        
        # Level 0 should have exactly one element
        if len(self.task_structure[0]) != 1:
            raise ValueError("First level of task_structure must have exactly one element (number of initial branches)")
        
        # Validate all values are positive
        for level_idx, level in enumerate(self.task_structure):
            if not all(isinstance(n, int) and n > 0 for n in level):
                raise ValueError(f"Level {level_idx} contains invalid values (must be positive integers)")
        
        # Validate consistency: each level must have the right number of elements
        for level_idx in range(1, len(self.task_structure)):
            prev_level = self.task_structure[level_idx - 1]
            curr_level = self.task_structure[level_idx]
            expected_length = sum(prev_level)  # Total branches from previous level
            
            if len(curr_level) != expected_length:
                raise ValueError(
                    f"Level {level_idx} has {len(curr_level)} elements, but should have "
                    f"{expected_length} (sum of level {level_idx - 1}: {prev_level})"
                )
        
        # Validate branch_layer_sizes if provided
        if self.branch_layer_sizes is not None:
            if len(self.branch_layer_sizes) != len(self.task_structure):
                raise ValueError(
                    f"branch_layer_sizes length ({len(self.branch_layer_sizes)}) must match "
                    f"task_structure length ({len(self.task_structure)})"
                )
            for i, layer_sizes in enumerate(self.branch_layer_sizes):
                if not layer_sizes or not all(s > 0 for s in layer_sizes):
                    raise ValueError(f"Invalid layer sizes at branch level {i}")
        
        # Validate shared layers if provided
        if self.shared_layer_sizes is not None:
            if not all(s > 0 for s in self.shared_layer_sizes):
                raise ValueError("All shared_layer_sizes must be positive")
        
        # Validate default layer sizes
        if not all(s > 0 for s in self.default_layer_sizes):
            raise ValueError("All default_layer_sizes must be positive")
        
        # Validate pooling strategy
        valid_pooling = {'concat', 'max', 'mean', 'sum', None}
        if self.pooling_strategy not in valid_pooling:
            raise ValueError(f"pooling_strategy must be one of {valid_pooling}")
        
        # Validate task_types if provided
        if self.task_types is not None:
            if len(self.task_types) != self.total_outputs:
                raise ValueError(
                    f"task_types length ({len(self.task_types)}) must equal "
                    f"total_outputs ({self.total_outputs})"
                )
            valid_types = {'binary', 'multiclass', 'multilabel'}
            for task_type in self.task_types:
                if task_type not in valid_types:
                    raise ValueError(
                        f"Invalid task_type '{task_type}'. Must be one of {valid_types}"
                    )
        
        # Validate num_classes_per_task if provided
        if self.num_classes_per_task is not None:
            if len(self.num_classes_per_task) != self.total_outputs:
                raise ValueError(
                    f"num_classes_per_task length ({len(self.num_classes_per_task)}) must equal "
                    f"total_outputs ({self.total_outputs})"
                )
            if not all(n >= 2 for n in self.num_classes_per_task):
                raise ValueError("All num_classes_per_task values must be >= 2")
            
            # If task_types is specified, validate consistency
            if self.task_types is not None:
                for task_idx, (task_type, num_classes) in enumerate(zip(self.task_types, self.num_classes_per_task)):
                    if task_type == 'binary' and num_classes != 2:
                        raise ValueError(
                            f"Task {task_idx} is type 'binary' but num_classes_per_task[{task_idx}]={num_classes}. "
                            f"Binary tasks should have num_classes=2."
                        )
                    elif task_type == 'multiclass' and num_classes < 3:
                        raise ValueError(
                            f"Task {task_idx} is type 'multiclass' but num_classes_per_task[{task_idx}]={num_classes}. "
                            f"Multiclass tasks should have num_classes >= 3."
                        )
        
        # Validate task_names if provided
        if self.task_names is not None:
            if len(self.task_names) != self.total_outputs:
                raise ValueError(
                    f"task_names length ({len(self.task_names)}) must equal "
                    f"total_outputs ({self.total_outputs})"
                )
    
    def get_layer_sizes(self, branch_level: int) -> List[int]:
        """Get layer sizes for a specific branch level.
        
        Args:
            branch_level: The branching level (0-indexed)
            
        Returns:
            List of layer sizes for that level
        """
        if self.branch_layer_sizes is not None:
            return self.branch_layer_sizes[branch_level]
        return self.default_layer_sizes
    
    @property
    def num_branch_levels(self) -> int:
        """Total number of branching levels in the hierarchy."""
        return len(self.task_structure)
    
    @property
    def total_outputs(self) -> int:
        """Total number of output predictions (final branches at last level)."""
        return sum(self.task_structure[-1])
    
    def get_branches_at_level(self, level: int) -> int:
        """Get the total number of branches at a specific level.
        
        Args:
            level: The branching level (0-indexed)
            
        Returns:
            Total number of branches at that level
        """
        return sum(self.task_structure[level])    
    def get_task_type(self, task_idx: int) -> str:
        """Get the type of a specific task.
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Task type: 'binary', 'multiclass', or 'multilabel'
        """
        if self.task_types is None:
            return 'binary'
        return self.task_types[task_idx]
    
    def get_num_classes(self, task_idx: int) -> int:
        """Get the number of classes for a specific task.
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Number of classes (2 for binary, K for multiclass)
        """
        if self.num_classes_per_task is None:
            return 2  # Default to binary
        return self.num_classes_per_task[task_idx]
    
    def get_output_dim_for_task(self, task_idx: int) -> int:
        """Get the output dimension for a specific task.
        
        Binary tasks output 1 dimension (probability of class 1).
        Multiclass tasks output K dimensions (probability distribution).
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Output dimension for this task
        """
        task_type = self.get_task_type(task_idx)
        num_classes = self.get_num_classes(task_idx)
        
        if task_type == 'binary':
            return 1
        elif task_type == 'multiclass' or task_type == 'multilabel':
            return num_classes
        else:
            raise ValueError(f"Unknown task_type: {task_type}")