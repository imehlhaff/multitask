"""Model saving and loading utilities."""

import json
import os
from typing import Optional, Union, Dict

import tensorflow as tf

from ..config import ModelConfig
from .hierarchical import MultiTaskModel


def load_model(
    filepath: str,
    encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
    custom_objects: Optional[dict] = None
) -> tuple[MultiTaskModel, Optional[Dict[str, Optional[float]]]]:
    """Load a saved MultiTaskModel from disk.
    
    Reconstructs complete model from saved artifacts including:
    - Model weights and architecture
    - Encoder configuration (TF Hub URL, HuggingFace name, or custom)
    - Optimized thresholds (if saved)
    
    Args:
        filepath: Directory path containing saved model artifacts
        encoder: Optional encoder override. If provided, uses this instead of saved encoder.
                 Useful for custom layers not serializable or updating encoder versions.
        custom_objects: Optional dict of custom objects for deserialization
        
    Returns:
        Tuple of (model, thresholds_dict or None)
        - model: Loaded MultiTaskModel ready for inference/training
        - thresholds: Dict mapping task_name -> float threshold, or None if not saved
        
    Raises:
        ValueError: If saved files are missing or invalid
        ImportError: If required dependencies not installed
        
    Example:
        >>> # Load model with thresholds
        >>> model, thresholds = load_model('my_model')
        >>> predictions = model.predict(test_data)
        >>> 
        >>> # Load with different encoder
        >>> custom_encoder = TFAutoModel.from_pretrained('bert-large')
        >>> model, thresholds = load_model('my_model', encoder=custom_encoder)
    """
    # Check that directory exists
    if not os.path.exists(filepath):
        raise ValueError(f"Model directory not found: {filepath}")
    
    # Load config
    config_path = os.path.join(filepath, 'config.json')
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_config = ModelConfig(**config_dict)
    
    # Load metadata
    metadata_path = os.path.join(filepath, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Determine encoder to use
    if encoder is None:
        # Use saved encoder
        if metadata['uses_encoder']:
            encoder_identifier = metadata.get('encoder_identifier')
            if encoder_identifier is None:
                raise ValueError(
                    "Model was saved with a custom encoder layer that cannot be automatically "
                    "loaded. Please provide the encoder explicitly via the encoder parameter."
                )
            encoder = encoder_identifier
        else:
            encoder = None
    
    # Create model with same architecture
    model = MultiTaskModel(
        config=model_config,
        encoder=encoder,
        embedding_dim=metadata.get('embedding_dim')
        # dropout_rate=metadata.get('dropout_rate', 0.3),
        # custom_objects=custom_objects
    )
    
    # Build model by calling it once (required before loading weights)
    # Create dummy input to infer shapes
    if model.uses_encoder:
        if model.encoder_type == 'huggingface':
            dummy_input = {
                'input_ids': tf.zeros((1, 128), dtype=tf.int32),
                'attention_mask': tf.ones((1, 128), dtype=tf.int32),
                'token_type_ids': tf.zeros((1, 128), dtype=tf.int32),
            }
        else:  # tfhub or custom
            dummy_input = {
                'input_word_ids': tf.zeros((1, 128), dtype=tf.int32),
                'input_mask': tf.ones((1, 128), dtype=tf.int32),
                'input_type_ids': tf.zeros((1, 128), dtype=tf.int32),
            }
    else:
        dummy_input = tf.zeros((1, metadata['embedding_dim']))
    
    _ = model(dummy_input, training=False)
    
    # Load weights
    weights_path = os.path.join(filepath, 'weights')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    elif os.path.exists(weights_path + '.h5'):
        model.load_weights(weights_path + '.h5')
    else:
        raise ValueError(f"Weights file not found: {weights_path}")
    
    # Load thresholds if available
    thresholds = None
    thresholds_path = os.path.join(filepath, 'thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
    
    print(f"Model loaded from {filepath}/")
    print(f"  - Encoder type: {model.encoder_type or 'embeddings'}")
    print(f"  - Tasks: {', '.join(model.config.task_names)}")
    print(f"  - Trainable parameters: {model.count_params():,}")
    if thresholds:
        print(f"  - Thresholds: {list(thresholds.keys())}")
    
    return model, thresholds
