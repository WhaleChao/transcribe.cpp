//! `serde` feature: the plain-data option and result types round-trip through
//! a human-readable format (JSON) and a binary one (postcard). The no-model
//! tier always runs; the model tier round-trips a real transcript.

#![cfg(feature = "serde")]

mod common;

use transcribe_cpp::{
    Backend, CommitPolicy, Diarize, Feature, Itn, KvType, Model, MoonshineStreamingOptions, Pnc,
    RunExtension, RunOptions, SessionOptions, SortformerPreset, SortformerStreamOptions,
    SpeakerSegment, StreamExtension, StreamOptions, StreamState, StreamText, StreamUpdate, Task,
    TimestampKind, Timings, Token, Transcript, WhisperRunOptions, Word,
};

fn json_round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let text = serde_json::to_string(value).expect("serialize json");
    serde_json::from_str(&text).expect("deserialize json")
}

fn postcard_round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = postcard::to_allocvec(value).expect("serialize postcard");
    postcard::from_bytes(&bytes).expect("deserialize postcard")
}

fn assert_round_trips<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(&json_round_trip(value), value, "json");
    assert_eq!(&postcard_round_trip(value), value, "postcard");
}

fn full_run_options() -> RunOptions {
    RunOptions {
        task: Task::Instruct,
        timestamps: TimestampKind::Word,
        pnc: Pnc::Off,
        itn: Itn::On,
        diarize: Diarize::On,
        language: Some("en".into()),
        target_language: Some("de".into()),
        keep_special_tags: true,
        spec_k_drafts: 4,
        family: Some(RunExtension::Whisper(WhisperRunOptions {
            initial_prompt: Some("hello".into()),
            temperature: Some(0.2),
            seed: Some(7),
            ..Default::default()
        })),
        vocabulary: vec!["transcribe.cpp".into(), "ggml".into()],
        prompt: Some("summarize".into()),
        prefix: Some("And so".into()),
    }
}

/// A transcript with every collection populated and a NaN confidence on both
/// NaN-able fields (JSON has no NaN; it travels as `null`).
fn full_transcript() -> Transcript {
    Transcript {
        text: "and so my fellow americans".into(),
        raw_text: " And so my fellow Americans".into(),
        language: Some("en".into()),
        timestamp_kind: TimestampKind::Token,
        segments: vec![Default::default()],
        speaker_segments: vec![SpeakerSegment {
            t0_ms: 0,
            t1_ms: 900,
            speaker_id: 1,
            p: f32::NAN,
        }],
        words: vec![Word {
            t0_ms: 10,
            t1_ms: 200,
            text: "and".into(),
            ..Default::default()
        }],
        tokens: vec![
            Token {
                id: 42,
                p: f32::NAN,
                text: " and".into(),
                ..Default::default()
            },
            Token {
                id: 43,
                p: 0.75,
                text: " so".into(),
                ..Default::default()
            },
        ],
        timings: Timings {
            load_ms: 1.0,
            mel_ms: 2.0,
            encode_ms: 3.0,
            decode_ms: 4.0,
        },
    }
}

/// NaN != NaN, so compare transcripts by their (deterministic) postcard
/// encoding, which carries the raw f32 bits.
fn same_transcript(a: &Transcript, b: &Transcript) -> bool {
    postcard::to_allocvec(a).unwrap() == postcard::to_allocvec(b).unwrap()
}

#[test]
fn run_options_round_trip() {
    assert_round_trips(&RunOptions::default());
    assert_round_trips(&full_run_options());
    let sortformer = RunOptions {
        family: Some(RunExtension::Sortformer(SortformerStreamOptions {
            preset: Some(SortformerPreset::LowLatency),
        })),
        ..Default::default()
    };
    assert_round_trips(&sortformer);
}

#[test]
fn stream_and_session_options_round_trip() {
    assert_round_trips(&StreamOptions::default());
    assert_round_trips(&StreamOptions {
        commit_policy: CommitPolicy::StablePrefix,
        stable_prefix_agreement_n: 2,
        family: Some(StreamExtension::MoonshineStreaming(
            MoonshineStreamingOptions {
                min_decode_interval_ms: Some(80),
            },
        )),
    });
    assert_round_trips(&SessionOptions {
        n_threads: 4,
        kv_type: KvType::F16,
        n_ctx: 448,
    });
}

#[test]
fn stream_results_and_enums_round_trip() {
    assert_round_trips(&StreamUpdate {
        result_changed: true,
        is_final: true,
        revision: 9,
        input_received_ms: 11_000,
        audio_committed_ms: 10_500,
        buffered_ms: 500,
        committed_changed: true,
        tentative_changed: false,
    });
    assert_round_trips(&StreamText {
        full: "a b c".into(),
        committed: "a b".into(),
        tentative: " c".into(),
    });
    assert_round_trips(&StreamState::Failed);
    assert_round_trips(&Backend::CpuAccel);
    assert_round_trips(&Feature::TranscriptPrefix);
}

#[test]
fn transcript_round_trips_nan_confidence() {
    let original = full_transcript();

    let json = serde_json::to_string(&original).unwrap();
    assert!(
        json.contains("\"p\":null"),
        "NaN should encode as null: {json}"
    );
    let from_json: Transcript = serde_json::from_str(&json).unwrap();
    assert!(from_json.tokens[0].p.is_nan());
    assert!(from_json.speaker_segments[0].p.is_nan());
    assert_eq!(from_json.tokens[1].p, 0.75);
    assert!(same_transcript(&from_json, &original), "json");

    let from_postcard = postcard_round_trip(&original);
    assert!(from_postcard.tokens[0].p.is_nan());
    assert!(same_transcript(&from_postcard, &original), "postcard");
}

#[test]
fn missing_fields_take_defaults() {
    // Option and result structs are `serde(default)`: a sparse document fills
    // the rest from `Default`, so a peer built against an older field set
    // still decodes.
    let run: RunOptions = serde_json::from_str(r#"{"language":"fr"}"#).unwrap();
    assert_eq!(
        run,
        RunOptions {
            language: Some("fr".into()),
            ..Default::default()
        }
    );
    let transcript: Transcript = serde_json::from_str(r#"{"text":"hi"}"#).unwrap();
    assert_eq!(transcript.text, "hi");
    assert!(transcript.segments.is_empty());
}

#[test]
fn real_transcript_round_trips() {
    let Some((model_path, pcm)) = common::smoke_fixtures("real_transcript_round_trips") else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut session = model.session().unwrap();
    let options = RunOptions {
        timestamps: TimestampKind::Token,
        ..Default::default()
    };
    let result = match session.run(&pcm, &options) {
        Ok(result) => result,
        // Fall back to the family's default granularity if token timestamps
        // are unsupported; the round-trip is what this test checks.
        Err(_) => session.run(&pcm, &RunOptions::default()).unwrap(),
    };
    assert!(!result.text.is_empty());

    assert!(same_transcript(&json_round_trip(&result), &result), "json");
    assert!(
        same_transcript(&postcard_round_trip(&result), &result),
        "postcard"
    );
    let caps = model.capabilities();
    assert_round_trips(&caps);
    assert_round_trips(&session.limits().unwrap());
}
