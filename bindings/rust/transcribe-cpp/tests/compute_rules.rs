//! Characterization of the rules every native compute call follows: the
//! per-model compute lock and stream lease, the order of the checks in front
//! of the native call (and therefore which error a racing caller sees), lease
//! ownership across sessions, cancel-token install/clear, and serialization of
//! run/run_batch across threads. These pin today's behavior so the compute
//! sites can be moved onto one helper without changing it. Model-gated: skip
//! cleanly when the canary GGUFs / jfk.wav are absent.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use transcribe_cpp::{
    CancelToken, Error, Feature, Model, RunOptions, StreamExtension, StreamOptions, StreamState,
};

const RUN_BUSY: &str = "a stream is active on this model; finish or drop it before run()";
const BATCH_BUSY: &str = "a stream is active on this model; finish or drop it before run_batch()";
const STREAM_BUSY: &str = "a stream is already active on this model";

fn streaming_fixtures(test: &str) -> Option<(PathBuf, Vec<f32>)> {
    match (common::smoke_streaming_model(), common::smoke_audio()) {
        (Some(m), Some(a)) => Some((m, a)),
        _ => {
            eprintln!("skip {test}: streaming model/audio absent");
            None
        }
    }
}

fn expect_busy<T: std::fmt::Debug>(r: transcribe_cpp::Result<T>, want: &str, what: &str) {
    match r {
        Err(Error::Busy(msg)) => assert_eq!(msg, want, "{what}: Busy message"),
        other => panic!("{what}: expected Error::Busy({want:?}), got {other:?}"),
    }
}

fn wrong_family_stream() -> StreamOptions {
    StreamOptions {
        family: Some(StreamExtension::ParakeetStream(Default::default())),
        ..Default::default()
    }
}

#[test]
fn busy_errors_name_the_refused_call() {
    // Each refusing site carries its own message: run and run_batch name the
    // call, a second stream says "already active".
    let Some((model_path, pcm)) = streaming_fixtures("busy_errors_name_the_refused_call") else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();

    let mut stream1 = s1
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    stream1.feed(&pcm[..pcm.len().min(1600)]).unwrap();

    expect_busy(s2.run(&pcm, &RunOptions::default()), RUN_BUSY, "run");
    expect_busy(
        s2.run_batch(&[&pcm], &RunOptions::default()),
        BATCH_BUSY,
        "run_batch",
    );
    expect_busy(
        s2.stream(&RunOptions::default(), &StreamOptions::default()),
        STREAM_BUSY,
        "stream",
    );
    // The refusals did not disturb the live stream.
    stream1.feed(&pcm[..pcm.len().min(1600)]).unwrap();
    assert_eq!(stream1.state(), StreamState::Active);
}

#[test]
fn option_validation_precedes_busy_check() {
    // Run options are marshalled before the compute lock is taken, so a bad
    // option is reported as itself even while another session streams.
    let Some((model_path, pcm)) = streaming_fixtures("option_validation_precedes_busy_check")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();
    let _stream1 = s1
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();

    let bad = RunOptions {
        language: Some("e\0n".into()),
        ..Default::default()
    };
    assert!(
        matches!(s2.run(&pcm, &bad), Err(Error::Nul(_))),
        "run: NUL option must win over Busy"
    );
    assert!(
        matches!(s2.run_batch(&[&pcm], &bad), Err(Error::Nul(_))),
        "run_batch: NUL option must win over Busy"
    );
    assert!(
        matches!(
            s2.stream(&bad, &StreamOptions::default()),
            Err(Error::Nul(_))
        ),
        "stream: NUL option must win over Busy"
    );
}

#[test]
fn busy_check_precedes_native_stream_validation() {
    // A begin that the native layer would reject (wrong-family extension) is
    // refused as Busy first when another session's stream holds the lease.
    let Some((model_path, pcm)) =
        streaming_fixtures("busy_check_precedes_native_stream_validation")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();

    let mut stream1 = s1
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    stream1.feed(&pcm[..pcm.len().min(1600)]).unwrap();
    expect_busy(
        s2.stream(&RunOptions::default(), &wrong_family_stream()),
        STREAM_BUSY,
        "wrong-family stream while busy",
    );
    drop(stream1);

    // With the lease free, the same begin reaches native validation.
    let err = s2
        .stream(&RunOptions::default(), &wrong_family_stream())
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "wrong-family stream when idle: {err:?}"
    );
}

#[test]
fn failed_stream_begin_does_not_take_the_lease() {
    let Some((model_path, _pcm)) =
        streaming_fixtures("failed_stream_begin_does_not_take_the_lease")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();

    assert!(s1
        .stream(&RunOptions::default(), &wrong_family_stream())
        .is_err());
    let stream2 = s2.stream(&RunOptions::default(), &StreamOptions::default());
    assert!(
        stream2.is_ok(),
        "a failed begin must leave the lease free: {stream2:?}"
    );
}

#[test]
fn ended_stream_never_releases_another_sessions_lease() {
    // After finalize() the lease belongs to whoever takes it next; a later
    // reset() or drop of the ended stream must not free that other lease.
    let Some((model_path, pcm)) =
        streaming_fixtures("ended_stream_never_releases_another_sessions_lease")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();
    let mut s3 = model.session().unwrap();
    let chunk = &pcm[..pcm.len().min(1600)];

    // finalize -> other session takes the lease -> ended stream reset()s.
    {
        let mut stream1 = s1
            .stream(&RunOptions::default(), &StreamOptions::default())
            .unwrap();
        stream1.feed(chunk).unwrap();
        stream1.finalize().unwrap();
        let stream2 = s2
            .stream(&RunOptions::default(), &StreamOptions::default())
            .unwrap();
        stream1.reset();
        expect_busy(
            s3.stream(&RunOptions::default(), &StreamOptions::default()),
            STREAM_BUSY,
            "after ended stream reset",
        );
        drop(stream2);
    }

    // finalize -> other session takes the lease -> ended stream is dropped.
    let mut stream1 = s1
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    stream1.feed(chunk).unwrap();
    stream1.finalize().unwrap();
    let stream2 = s2
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    drop(stream1);
    expect_busy(
        s3.stream(&RunOptions::default(), &StreamOptions::default()),
        STREAM_BUSY,
        "after ended stream drop",
    );
    expect_busy(s3.run(&pcm, &RunOptions::default()), RUN_BUSY, "run");

    // Releasing the real holder frees the model.
    drop(stream2);
    let stream3 = s3
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    assert_eq!(stream3.state(), StreamState::Active);
}

#[test]
fn run_and_run_batch_across_threads_serialize() {
    // One-shot runs and batches on different sessions of one model queue on
    // the compute lock (never Busy), and each gets its own results back.
    let Some((model_path, pcm)) =
        common::smoke_fixtures("run_and_run_batch_across_threads_serialize")
    else {
        return;
    };
    let model = Arc::new(Model::load(&model_path).unwrap());
    let pcm = Arc::new(pcm);
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let model = Arc::clone(&model);
            let pcm = Arc::clone(&pcm);
            thread::spawn(move || {
                let mut s = model.session().unwrap();
                if i % 2 == 0 {
                    vec![s.run(&pcm, &RunOptions::default()).unwrap().text]
                } else {
                    s.run_batch(&[&pcm, &pcm], &RunOptions::default())
                        .unwrap()
                        .into_iter()
                        .map(|r| r.unwrap().text)
                        .collect()
                }
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let texts = h.join().unwrap();
        assert_eq!(texts.len(), if i % 2 == 0 { 1 } else { 2 });
        for t in texts {
            assert!(t.to_lowercase().contains("country"), "thread {i}: {t}");
        }
    }
}

#[test]
fn cancel_token_is_retained_until_cleared() {
    // The session keeps the installed token's flag alive (the caller may drop
    // every handle) and keeps consulting it on later runs until cleared.
    let Some((model_path, pcm)) = common::smoke_fixtures("cancel_token_is_retained_until_cleared")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    if !model.supports(Feature::Cancellation) {
        eprintln!("skip: model does not support cancellation");
        return;
    }
    let mut session = model.session().unwrap();
    {
        let token = CancelToken::new();
        token.cancel();
        session.set_cancel_token(&token);
    } // every caller-side handle to the flag is gone

    for attempt in 0..2 {
        match session.run(&pcm, &RunOptions::default()) {
            Err(Error::Aborted { .. }) => assert!(session.was_aborted()),
            other => panic!("attempt {attempt}: pre-cancelled token must abort, got {other:?}"),
        }
    }

    session.clear_cancel_token();
    let result = session.run(&pcm, &RunOptions::default()).unwrap();
    assert!(result.text.to_lowercase().contains("country"));
    assert!(!session.was_aborted());
}
