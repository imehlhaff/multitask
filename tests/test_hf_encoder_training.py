"""
Regression tests: a Hugging Face encoder must actually fine-tune, be restored by early stopping, and be saved.

transformers builds TFAutoModel with tf-keras (Keras 2). A Keras 3 MultiTaskModel doesn't track the weights of a
tf-keras model stored as an attribute, so the encoder's weights used to get no gradient updates, weren't restored by
EarlyStopping(restore_best_weights=True), and weren't saved. Training still ran and produced (poor) numbers, so nothing
failed. These tests use a real tiny Hugging Face model, since a mock or a Keras 3 stand-in encoder can't reproduce the
tracking gap.

They run when the model is in the local Hugging Face cache, or with MULTITASK_RUN_HF_FROM_PT_INTEGRATION=1 to download
it.
"""
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("transformers")

from multitask import MultiTaskModel, Trainer, TrainingConfig, load_model  # noqa: E402
from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig  # noqa: E402

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"


def _model_available():
    if os.getenv("MULTITASK_RUN_HF_FROM_PT_INTEGRATION", "0") == "1":
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(MODEL_ID, "config.json"), str)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _model_available(),
        reason=f"{MODEL_ID} isn't cached; set MULTITASK_RUN_HF_FROM_PT_INTEGRATION=1 to download it",
    ),
]


def _config(trainable_encoder=True):
    return ModelConfig(
        task_structure=[[1]],
        task_names=["t"],
        task_types=["binary"],
        num_classes_per_task=[2],
        encoder_config=EncoderConfig(
            encoder_identifier=MODEL_ID,
            encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            # The tiny BERTs only ship PyTorch weights
            from_pt=True,
            trainable_encoder=trainable_encoder,
        ),
    )


def _data(n=64, seq_len=8):
    """Random token ids, with the label given away by the token in position 1."""
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1000, 2000, size=(n, seq_len)).astype("int32")
    y = (np.arange(n) % 2).astype("float32")
    input_ids[:, 1] = np.where(y == 1, 2204, 2919)
    return {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}, y


def _dataset():
    X, y = _data()
    return tf.data.Dataset.from_tensor_slices((X, {"t": y})).batch(16)


def _build(trainable_encoder=True):
    tf.keras.utils.set_random_seed(0)
    model = MultiTaskModel(_config(trainable_encoder))
    model(_data()[0])
    return model


def _encoder_values(model):
    return [v.numpy().copy() for v in model.encoder.weights]


def _fit(model):
    trainer = Trainer(model, TrainingConfig(epochs=1, batch_size=16, learning_rate=1e-3))
    trainer.fit(_dataset(), _dataset(), num_tasks=1)


def test_every_encoder_weight_is_trained():
    """Each trainable encoder weight is either tracked by Keras or added by _untracked_encoder_weights."""
    model = _build()
    trained = {id(v) for v in model.trainable_variables}
    trained |= {id(v) for v in model._untracked_encoder_weights(trainable_only=True)}
    missing = [v.name for v in model.encoder.trainable_weights if id(v) not in trained]
    assert model.encoder.trainable_weights, "the encoder has no trainable weights to check"
    assert not missing, f"encoder weights that would never be updated: {missing[:5]}"


def test_fit_updates_encoder_weights():
    model = _build()
    before = _encoder_values(model)
    _fit(model)
    changed = [not np.allclose(b, a) for b, a in zip(before, _encoder_values(model))]
    assert any(changed), "training left every encoder weight unchanged"


def test_frozen_encoder_stays_frozen():
    model = _build(trainable_encoder=False)
    before = _encoder_values(model)
    _fit(model)
    for b, a in zip(before, _encoder_values(model)):
        np.testing.assert_array_equal(b, a)


def test_get_and_set_weights_include_encoder():
    """EarlyStopping(restore_best_weights=True) snapshots get_weights() and restores it with set_weights()."""
    model = _build()
    snapshot = model.get_weights()
    original = _encoder_values(model)
    for v in model.encoder.weights:
        v.assign(v + 1.0)
    model.set_weights(snapshot)
    for o, r in zip(original, _encoder_values(model)):
        np.testing.assert_array_equal(o, r)


def test_save_and_load_keep_fine_tuned_encoder(tmp_path):
    model = _build()
    rng = np.random.default_rng(1)
    # Stand-in for fine-tuning: move every encoder weight away from its pretrained value
    for v in model.encoder.weights:
        v.assign(v + 0.01 * rng.standard_normal(v.shape).astype(v.dtype.as_numpy_dtype))
    X, _ = _data()
    expected = model(X, training=False)["t"].numpy()

    model.save(str(tmp_path / "model"))
    loaded, _ = load_model(str(tmp_path / "model"))

    for o, r in zip(_encoder_values(model), _encoder_values(loaded)):
        np.testing.assert_allclose(o, r)
    np.testing.assert_allclose(loaded(X, training=False)["t"].numpy(), expected, rtol=1e-5, atol=1e-6)
