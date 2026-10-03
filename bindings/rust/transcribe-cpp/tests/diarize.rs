//! DIARIZE role: roles, diarize_info, DiarizeSession runs, and refusal on an
//! ASR-only model. Model-gated: Sortformer on TRANSCRIBE_SMOKE_SORTFORMER_MODEL
//! (+ samples/sortformer-2spk-mix.wav), whisper on TRANSCRIBE_SMOKE_MODEL.
//! The Busy path is covered model-free in `src/diarize.rs` (no model serves
//! both DIARIZE and streaming today).

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE as SFDR;
use transcribe_cpp::{
    Backend, CancelToken, DiarizeExtension, DiarizeOptions, Error, ExtSlot, Model, ModelOptions,
    Role, SortformerDiarizeOptions, SortformerPreset, SpeakerSegment,
};

fn diarize_opts(preset: Option<SortformerPreset>) -> DiarizeOptions {
    DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset,
        })),
    }
}

fn rows(segs: &[SpeakerSegment]) -> Vec<(i64, i64, i32)> {
    segs.iter()
        .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id))
        .collect()
}

#[test]
fn sortformer_is_diarize_only() {
    let Some((model_path, _)) = common::smoke_sortformer_fixtures("sortformer_is_diarize_only")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(!roles.contains(Role::Asr) && roles.contains(Role::Diarize));
    assert_eq!(roles.bits(), 0b10, "{roles:?}");
    let info = model.diarize_info().unwrap();
    assert_eq!((info.sample_rate, info.max_speakers), (16000, 4));
    assert!(model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
    assert!(!model.accepts_ext(ExtSlot::Run, SFDR));
    assert!(matches!(
        model.capabilities(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(model.session(), Err(Error::UnsupportedRole(_))));
}

#[test]
fn diarize_run_matches_cpu_golden_segments() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_run_matches_cpu_golden_segments")
    else {
        return;
    };
    // Goldens are recorded on CPU (tests/sortformer_diarize_unit.cpp).
    let cpu = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    let model = Model::load_with(&model_path, &cpu).unwrap();
    let mut diarize = model.diarize_session().unwrap();
    let golden_default = [
        (320, 2400, 1),
        (7360, 9360, 1),
        (10240, 10640, 1),
        (4240, 6640, 2),
        (9760, 12000, 2),
    ];
    let golden_low_latency = [
        (320, 2480, 1),
        (7360, 9360, 1),
        (10240, 10640, 1),
        (4160, 6640, 2),
        (9760, 12000, 2),
    ];
    for (preset, golden) in [
        (None, &golden_default),
        (Some(SortformerPreset::Default), &golden_default),
        (Some(SortformerPreset::LowLatency), &golden_low_latency),
    ] {
        let turns = diarize.run(&pcm, &diarize_opts(preset)).unwrap();
        assert_eq!(rows(&turns), golden.to_vec(), "{preset:?}");
        assert!(diarize.timings().encode_ms > 0.0);
    }
}

#[test]
fn diarize_session_keeps_model_alive_and_rejects_bad_input() {
    let Some((model_path, pcm)) = common::smoke_sortformer_fixtures(
        "diarize_session_keeps_model_alive_and_rejects_bad_input",
    ) else {
        return;
    };
    let mut diarize = Model::load(&model_path).unwrap().diarize_session().unwrap();
    let opts = DiarizeOptions::default();
    for bad in [&[][..], &[f32::NAN; 16][..]] {
        let err = diarize.run(bad, &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }
    assert!(!diarize.run(&pcm, &opts).unwrap().is_empty());
}

#[test]
fn cancelled_diarize_run_is_aborted() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("cancelled_diarize_run_is_aborted")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut diarize = model.diarize_session().unwrap();
    let token = CancelToken::new();
    diarize.set_cancel_token(&token);
    token.cancel();
    let err = diarize.run(&pcm, &DiarizeOptions::default()).unwrap_err();
    assert!(matches!(err, Error::Aborted { .. }), "{err:?}");
    diarize.clear_cancel_token();
    assert!(!diarize
        .run(&pcm, &DiarizeOptions::default())
        .unwrap()
        .is_empty());
}

#[test]
fn asr_only_model_refuses_diarize() {
    let Some(model_path) = common::smoke_model() else {
        eprintln!("skip asr_only_model_refuses_diarize: smoke model absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(roles.contains(Role::Asr) && !roles.contains(Role::Diarize));
    assert!(matches!(
        model.diarize_info(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(
        model.diarize_session(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(!model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
}
