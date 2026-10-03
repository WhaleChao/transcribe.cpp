//! DIARIZE role on Sortformer (TRANSCRIBE_SMOKE_SORTFORMER_MODEL) and whisper
//! (TRANSCRIBE_SMOKE_MODEL); the Busy path is covered in `src/diarize.rs`.

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE as SFDR;
use transcribe_cpp::{
    Backend, CancelToken, DiarizeExtension, DiarizeOptions, Error, ExtSlot, Model, ModelOptions,
    Role, SortformerDiarizeOptions, SortformerPreset,
};

#[test]
fn sortformer_is_diarize_only() {
    let Some((model_path, _)) = common::smoke_sortformer_fixtures("sortformer_is_diarize_only")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(!roles.contains(Role::Asr) && roles.contains(Role::Diarize));
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
        let opts = DiarizeOptions {
            family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                preset,
            })),
        };
        let turns = diarize.run(&pcm, &opts).unwrap();
        let rows: Vec<_> = turns
            .iter()
            .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id))
            .collect();
        assert_eq!(rows, golden.to_vec(), "{preset:?}");
        assert!(diarize.timings().encode_ms > 0.0);
    }
}

#[test]
fn diarize_session_rejects_bad_input_and_cancels() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_session_rejects_bad_input_and_cancels")
    else {
        return;
    };
    // The session keeps its model alive after the last Model handle drops.
    let mut diarize = Model::load(&model_path).unwrap().diarize_session().unwrap();
    let opts = DiarizeOptions::default();
    for bad in [&[][..], &[f32::NAN; 16][..]] {
        let err = diarize.run(bad, &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }
    let token = CancelToken::new();
    diarize.set_cancel_token(&token);
    token.cancel();
    let err = diarize.run(&pcm, &opts).unwrap_err();
    assert!(matches!(err, Error::Aborted { .. }), "{err:?}");
    diarize.clear_cancel_token();
    assert!(!diarize.run(&pcm, &opts).unwrap().is_empty());
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
