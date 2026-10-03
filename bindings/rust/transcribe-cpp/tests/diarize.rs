//! DIARIZE role: roles, diarize_info, DiarizeSession runs, and refusal on an
//! ASR-only model. Model-gated: Sortformer on TRANSCRIBE_SMOKE_SORTFORMER_MODEL
//! (+ samples/sortformer-2spk-mix.wav), whisper on TRANSCRIBE_SMOKE_MODEL.
//! The Busy path is covered model-free in `src/diarize.rs` (no model serves
//! both DIARIZE and streaming today).

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE as SFDR;
use transcribe_cpp::{
    CancelToken, DiarizeExtension, DiarizeOptions, Error, ExtSlot, Model, Role, RunExtension,
    RunOptions, SortformerDiarizeOptions, SortformerPreset, SortformerStreamOptions,
    SpeakerSegment,
};

fn diarize_opts(preset: Option<SortformerPreset>) -> DiarizeOptions {
    DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset,
        })),
    }
}

fn rows(segs: &[SpeakerSegment]) -> Vec<(i64, i64, i32, u32)> {
    segs.iter()
        .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id, s.p.to_bits()))
        .collect()
}

#[test]
fn sortformer_serves_asr_and_diarize() {
    let Some((model_path, _)) =
        common::smoke_sortformer_fixtures("sortformer_serves_asr_and_diarize")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(roles.contains(Role::Asr) && roles.contains(Role::Diarize));
    assert_eq!(roles.bits(), 0b11, "{roles:?}");
    let info = model.diarize_info().unwrap();
    assert_eq!((info.sample_rate, info.max_speakers), (16000, 4));
    assert!(model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
    assert!(!model.accepts_ext(ExtSlot::Run, SFDR));
    assert!(model.capabilities().is_ok());
}

#[test]
fn diarize_run_matches_asr_path_speaker_turns() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_run_matches_asr_path_speaker_turns")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut session = model.session().unwrap();
    let mut diarize = model.diarize_session().unwrap();
    for preset in [None, Some(SortformerPreset::LowLatency)] {
        let turns = diarize.run(&pcm, &diarize_opts(preset)).unwrap();
        assert!(!turns.is_empty(), "{preset:?}: no speaker turns");
        assert!(turns.iter().all(|s| (1..=4).contains(&s.speaker_id)));
        assert!(diarize.timings().encode_ms > 0.0);

        // PARITY with the ASR path (Session::run + SFST preset). The ASR path
        // for Sortformer is removed in a later commit; drop this block then.
        let asr = RunOptions {
            family: Some(RunExtension::Sortformer(SortformerStreamOptions { preset })),
            ..Default::default()
        };
        let transcript = session.run(&pcm, &asr).unwrap();
        assert_eq!(
            rows(&turns),
            rows(&transcript.speaker_segments),
            "{preset:?}"
        );
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
