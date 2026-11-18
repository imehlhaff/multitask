"""Model saving and loading utilities."""

import json
import os
import warnings
from typing import Optional, Union, Dict

import tensorflow as tf

from ..config import ModelConfig, EncoderConfig, EncoderInputType, EncoderIntegration
from .hierarchical import MultiTaskModel


def _get_build_dummy_input(model: MultiTaskModel, metadata: dict):
    """Build a dummy input suitable for building the model (one forward pass) before loading weights.

    Chooses input format based on encoder type so that model(dummy_input) succeeds:
    - No encoder: pre-computed embeddings, shape (1, embedding_dim)
    - HuggingFace: dict with input_ids, attention_mask, token_type_ids, each (1, 128)
    - TF Hub Universal Sentence Encoder: string tensor
    - TF Hub BERT-style: dict with input_word_ids, input_mask, input_type_ids
    - Custom encoder: string tensor (common default; user may need to provide encoder if this fails)

    Args:
        model: MultiTaskModel instance (uses_encoder, encoder_type set).
        metadata: Saved metadata dict (encoder_identifier, embedding_dim).

    Returns:
        Dummy input (Tensor or dict of Tensors) for model(dummy_input, training=False).
    """
    meta = metadata or {}
    uses_encoder = getattr(model, "uses_encoder", True)

    def _embedding_dim() -> int:
        if meta.get("embedding_dim") is not None:
            return int(meta["embedding_dim"])
        cfg = getattr(model, "config", None)
        ec = getattr(cfg, "encoder_config", None) if cfg is not None else None
        if ec is not None and ec.embedding_dim is not None:
            return int(ec.embedding_dim)
        ed = getattr(model, "embedding_dim", None)
        if ed is not None:
            return int(ed)
        return 128

    input_type = getattr(model, "encoder_input_type", None)
    if input_type is None:
        enc_type = getattr(model, "encoder_type", None)
        if not uses_encoder:
            input_type = EncoderInputType.PRECOMPUTED
        elif enc_type == "huggingface":
            input_type = EncoderInputType.HUGGINGFACE_TOKENS
        elif enc_type == "tfhub":
            ident = (meta.get("encoder_identifier") or "").lower()
            if "universal-sentence-encoder" in ident:
                input_type = EncoderInputType.RAW_STRING
            else:
                input_type = EncoderInputType.TFHUB_TOKENS
        elif enc_type in (
            "openai",
            "cohere",
            "voyage",
            "huggingface_api",
            "keras",
            "custom",
        ):
            input_type = EncoderInputType.RAW_STRING
        else:
            input_type = EncoderInputType.PRECOMPUTED

    if input_type == EncoderInputType.HUGGINGFACE_TOKENS:
        return {
            "input_ids": tf.zeros((1, 128), dtype=tf.int32),
            "attention_mask": tf.ones((1, 128), dtype=tf.int32),
            "token_type_ids": tf.zeros((1, 128), dtype=tf.int32),
        }
    if input_type == EncoderInputType.TFHUB_TOKENS:
        return {
            "input_word_ids": tf.zeros((1, 128), dtype=tf.int32),
            "input_mask": tf.ones((1, 128), dtype=tf.int32),
            "input_type_ids": tf.zeros((1, 128), dtype=tf.int32),
        }
    if input_type == EncoderInputType.RAW_STRING:
        return tf.constant(["dummy text"], dtype=tf.string)
    return tf.zeros((1, _embedding_dim()))


def load_model(
    filepath: str,
    encoder: Optional[Union[str, tf.keras.layers.Layer]] = None
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
        
    Returns:
        Tuple of (model, thresholds_dict or None)
        - model: Loaded MultiTaskModel ready for inference/training
        - thresholds: Dict mapping output name -> threshold (float or None), or None if not saved
        
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

    # Support nested encoder_config or flat encoder_* keys (backward compat)
    if "encoder_config" in config_dict:
        encoder_config = EncoderConfig.from_dict(config_dict.pop("encoder_config"))
        model_config = ModelConfig(encoder_config=encoder_config, **config_dict)
    else:
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
        embedding_dim=metadata.get('embedding_dim'),
        dropout_rate=metadata.get('dropout_rate', 0.3)
    )
    
    # Build model by calling it once (required before loading weights)
    dummy_input = _get_build_dummy_input(model, metadata)
    _ = model(dummy_input, training=False)
    
    # Load weights (Keras 3 uses .weights.h5; support legacy paths for backward compat)
    weights_path_h5 = os.path.join(filepath, 'weights.weights.h5')
    weights_path_legacy = os.path.join(filepath, 'weights')
    has_new = os.path.exists(weights_path_h5)
    has_index = os.path.exists(weights_path_legacy + '.index')
    has_direct = os.path.exists(weights_path_legacy)
    has_h5 = os.path.exists(weights_path_legacy + '.h5')

    if has_new:
        model.load_task_weights(weights_path_h5)
    elif has_index or has_direct or has_h5:
        model.load_task_weights(weights_path_legacy)
    else:
        raise ValueError(f"Weights file not found: {filepath} (expected weights.weights.h5 or legacy weights/weights.h5)")

    # A Hugging Face encoder's weights aren't in weights.weights.h5; MultiTaskModel.save writes them separately
    encoder_weights_path = os.path.join(filepath, 'encoder_weights.npz')
    if os.path.exists(encoder_weights_path):
        model.load_untracked_encoder_weights(encoder_weights_path)
    elif model._untracked_encoder_weights():
        warnings.warn(
            f"{filepath} has no encoder_weights.npz, so the encoder keeps its pretrained weights. Models saved before "
            "Hugging Face encoder weights were saved separately never stored their fine-tuned encoder.",
            stacklevel=2,
        )

    # Load thresholds if available
    thresholds = None
    thresholds_path = os.path.join(filepath, 'thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
            
    # Load regression standardization stats if available
    regression_stats_path = os.path.join(filepath, 'regression_stats.json')
    if os.path.exists(regression_stats_path):
        with open(regression_stats_path, 'r') as f:
            model._regression_stats = json.load(f)
    else:
        model._regression_stats = {}
    
    print(f"Model loaded from {filepath}/")
    print(f"  - Encoder type: {model.encoder_type or 'embeddings'}")
    task_names = model.config.task_names or [f'task_{i}' for i in range(model.config.total_outputs)]
    print(f"  - Tasks: {', '.join(task_names)}")
    if model.config.has_pools:
        print(f"  - Output pools: {', '.join(model.config.pool_names)}")
    print(f"  - Trainable parameters: {model.count_params():,}")
    if thresholds:
        print(f"  - Thresholds: {list(thresholds.keys())}")
    
    return model, thresholds