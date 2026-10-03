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
        assert model.roles == {t.Role.ASR, t.Role.DIARIZE}
        assert model.diarize_info == t.DiarizeInfo(sample_rate=16000, max_speakers=4)


def test_asr_only_model_rejects_diarize(model_path):
    with t.Model(model_path) as model:
        assert model.roles == {t.Role.ASR}
        with pytest.raises(t.UnsupportedRole):
            model.diarize_info
        with pytest.raises(t.UnsupportedRole):
            model.diarize_session()


# TEMPORARY: Sortformer still serves the ASR role, whose Session.run returns
# the same speaker turns on Result.speaker_segments. This cross-check goes
# away when that ASR path is removed in a later commit.
@pytest.mark.parametrize("preset", ["default", "low_latency"])
def test_diarize_run_matches_asr_path(sortformer_model_path, mix_pcm, preset):
    with t.Model(sortformer_model_path) as model:
        with model.diarize_session() as d:
            rows = d.run(mix_pcm, family=t.SortformerDiarizeOptions(preset=preset))
            timings = d.timings
        with model.session() as s:
            asr = s.run(mix_pcm, family=t.SortformerStreamOptions(preset=preset))
            with pytest.raises(t.InvalidArgument, match="slot"):
                s.run(mix_pcm, family=t.SortformerDiarizeOptions())
    assert rows and isinstance(rows, list)
    assert all(isinstance(r, t.SpeakerSegment) for r in rows)
    assert {r.speaker_id for r in rows} <= {1, 2, 3, 4}
    assert all(math.isnan(r.p) for r in rows)  # Sortformer has no per-turn p
    assert _turns(rows) == _turns(asr.speaker_segments)
    assert timings.encode_ms > 0


def test_diarize_run_without_extension(sortformer_model_path, mix_pcm):
    with t.Model(sortformer_model_path) as model, model.diarize_session() as d:
        assert _turns(d.run(mix_pcm)) == _turns(
            d.run(mix_pcm, family=t.SortformerDiarizeOptions(preset="default")))


def test_bad_preset_rejected(sortformer_model_path, mix_pcm):
    class OutOfRange(t.SortformerDiarizeOptions):
        _presets = {**t.SortformerDiarizeOptions._presets, "bogus": 99}

    with pytest.raises(ValueError, match="preset"):
        t.SortformerDiarizeOptions(preset="bogus")  # type: ignore[arg-type]
    with t.Model(sortformer_model_path) as model, model.diarize_session() as d:
        with pytest.raises(t.InvalidArgument):
            d.run(mix_pcm, family=OutOfRange(preset="bogus"))  # type: ignore[arg-type]
        with pytest.raises(t.InvalidArgument, match="slot"):
            d.run(mix_pcm, family=t.SortformerStreamOptions())
        assert d.run(mix_pcm)  # still usable


def test_diarize_session_keeps_model_alive(sortformer_model_path, mix_pcm):
    d = t.Model(sortformer_model_path).diarize_session()
    gc.collect()
    assert d.run(mix_pcm)
    d.close()
    with pytest.raises(t.TranscribeError, match="closed"):
        d.run(mix_pcm)
