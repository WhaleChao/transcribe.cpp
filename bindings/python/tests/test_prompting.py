"""Generic prompting inputs: run-params marshalling and model-gated behavior."""

import ctypes

import pytest

import transcribe_cpp as t
from transcribe_cpp import _generated


def test_run_params_carry_prompting_fields():
    params = t._build_run_params("instruct", None, None, "none", False, -1,
                                 vocabulary=["GGUF", "ggml"], prompt="Summarize.",
                                 prefix="And so")
    assert params.task == _generated.TRANSCRIBE_TASK_INSTRUCT
    assert params.n_vocabulary == 2
    assert [params.vocabulary[i] for i in range(2)] == [b"GGUF", b"ggml"]
    assert params.prompt == b"Summarize."
    assert params.prefix == b"And so"


def test_run_params_default_to_no_prompting():
    params = t._build_run_params("transcribe", None, None, "auto", False, -1)
    assert params.n_vocabulary == 0
    assert not params.vocabulary
    assert params.prompt is None and params.prefix is None


def test_vocabulary_rejects_single_string():
    with pytest.raises(t.InvalidArgument):
        t._build_run_params("transcribe", None, None, "auto", False, -1, vocabulary="GGUF")


def test_prompting_features_probe(model_path):
    with t.Model(model_path, backend="cpu") as model:
        for feature in ("vocabulary", "context_prompt", "instruct", "transcript_prefix"):
            assert isinstance(model.supports(feature), bool)


def test_unsupported_prefix_raises(model_path, audio_pcm):
    with t.Model(model_path, backend="cpu") as model, model.session() as session:
        if model.supports("transcript_prefix"):
            pytest.skip("model supports a transcript prefix")
        with pytest.raises(t.InvalidArgument):
            session.run(audio_pcm, prefix="And so")
