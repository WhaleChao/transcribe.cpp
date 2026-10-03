"""DIARIZE role against real models: roles, diarize info, DiarizeSession.run
and its Sortformer preset extension. The locking / Busy / close / cancel
rules are pinned model-free in test_compute_lock.py."""

from __future__ import annotations

import gc
import math

import pytest

import transcribe_cpp as t
from conftest import SAMPLES, load_wav


@pytest.fixture(scope="module")
def mix_pcm():
    return load_wav(SAMPLES / "sortformer-2spk-mix.wav")


def _turns(rows):
    return [(r.t0_ms, r.t1_ms, r.speaker_id) for r in rows]


def test_sortformer_roles_and_info(sortformer_model_path):
    with t.Model(sortformer_model_path) as model:
        assert model.roles == {t.Role.DIARIZE}
        assert model.diarize_info == t.DiarizeInfo(sample_rate=16000, max_speakers=4)


def test_sortformer_rejects_asr(sortformer_model_path):
    with t.Model(sortformer_model_path) as model:
        with pytest.raises(t.UnsupportedRole):
            model.capabilities
        with pytest.raises(t.UnsupportedRole):
            model.session()
        with model.diarize_session():  # the DIARIZE role still opens
            pass


def test_asr_only_model_rejects_diarize(model_path):
    with t.Model(model_path) as model:
        assert model.roles == {t.Role.ASR}
        with pytest.raises(t.UnsupportedRole):
            model.diarize_info
        with pytest.raises(t.UnsupportedRole):
            model.diarize_session()


# CPU goldens on the oracle mix, (t0_ms, t1_ms, speaker_id) grouped by speaker.
GOLDEN = {
    "default": [(320, 2400, 1), (7360, 9360, 1), (10240, 10640, 1),
                (4240, 6640, 2), (9760, 12000, 2)],
    "low_latency": [(320, 2480, 1), (7360, 9360, 1), (10240, 10640, 1),
                    (4160, 6640, 2), (9760, 12000, 2)],
}


@pytest.mark.parametrize("preset", sorted(GOLDEN))
def test_diarize_run_golden_segments(sortformer_model_path, mix_pcm, preset):
    with t.Model(sortformer_model_path, backend="cpu") as model:
        with model.diarize_session() as d:
            rows = d.run(mix_pcm, family=t.SortformerDiarizeOptions(preset=preset))
            timings = d.timings
    assert all(math.isnan(r.p) for r in rows)  # Sortformer has no per-turn p
    assert _turns(rows) == GOLDEN[preset]
    assert timings.encode_ms > 0


def test_diarize_run_without_extension(sortformer_model_path, mix_pcm):
    with t.Model(sortformer_model_path) as model, model.diarize_session() as d:
        assert _turns(d.run(mix_pcm)) == _turns(
            d.run(mix_pcm, family=t.SortformerDiarizeOptions(preset="default")))


def test_bad_preset_rejected(sortformer_model_path, mix_pcm):
    class OutOfRange(t.SortformerDiarizeOptions):
        _presets = {**t.SortformerDiarizeOptions._presets, "bogus": 99}

    with t.Model(sortformer_model_path) as model, model.diarize_session() as d:
        with pytest.raises(t.InvalidArgument):
            d.run(mix_pcm, family=OutOfRange(preset="bogus"))  # type: ignore[arg-type]
        with pytest.raises(t.InvalidArgument, match="slot"):
            d.run(mix_pcm, family=t.WhisperRunOptions())
        assert d.run(mix_pcm)  # still usable


def test_diarize_session_keeps_model_alive(sortformer_model_path, mix_pcm):
    d = t.Model(sortformer_model_path).diarize_session()
    gc.collect()
    assert d.run(mix_pcm)
    d.close()
    with pytest.raises(t.TranscribeError, match="closed"):
        d.run(mix_pcm)
