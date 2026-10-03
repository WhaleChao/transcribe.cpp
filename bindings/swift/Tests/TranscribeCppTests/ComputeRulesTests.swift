import Foundation
import XCTest

@testable import TranscribeCpp

/// Characterization of the rules every native compute call follows today, so
/// moving them behind one helper can be checked against the old behavior:
///
/// - every compute site (run, runBatch, stream begin/feed/finalize/reset, and a
///   dropped `Stream`'s reset) waits on the model-wide `runLock`, which all
///   sessions of one model share;
/// - run / runBatch / stream begin validate their string options BEFORE taking
///   the lock, and refuse an active stream with `.busy` only AFTER taking it;
/// - the async bridge installs its own cancel token only when none is set, and
///   removes it when the call returns (including after an abort);
/// - a `Session` keeps its `Model` alive and a `Stream` keeps its `Session`
///   alive, so dropping the caller's references mid-use is safe;
/// - results are owned Swift copies that later compute cannot disturb.
final class ComputeRulesTests: XCTestCase {
    // MARK: - Helpers

    private final class Box<T> { var value: T? }

    /// Hold `model.runLock` on this thread, run `op` on a background queue,
    /// and require it to still be blocked after a short wait; then release the
    /// lock and require it to finish. Bounded waits, so a regression fails
    /// rather than hanging the suite.
    private func assertWaitsOnModelLock(
        _ model: Model, _ what: String,
        file: StaticString = #filePath, line: UInt = #line,
        _ op: @escaping () -> Void
    ) {
        let done = DispatchSemaphore(value: 0)
        model.runLock.lock()
        var held = true
        defer { if held { model.runLock.unlock() } }
        DispatchQueue.global().async { op(); done.signal() }
        let early = done.wait(timeout: .now() + 0.3)
        XCTAssertEqual(
            early, .timedOut, "\(what) must wait on the model compute lock",
            file: file, line: line)
        model.runLock.unlock()
        held = false
        if early == .success { return }
        XCTAssertEqual(
            done.wait(timeout: .now() + 60), .success,
            "\(what) must proceed once the model lock is free", file: file, line: line)
    }

    /// Hold `model.runLock` on this thread and require `op` (on a background
    /// queue) to finish anyway: it must not touch the lock.
    private func assertSkipsModelLock(
        _ model: Model, _ what: String,
        file: StaticString = #filePath, line: UInt = #line,
        _ op: @escaping () -> Void
    ) {
        let done = DispatchSemaphore(value: 0)
        model.runLock.lock()
        DispatchQueue.global().async { op(); done.signal() }
        let result = done.wait(timeout: .now() + 10)
        model.runLock.unlock()
        XCTAssertEqual(
            result, .success, "\(what) must not wait on the model compute lock",
            file: file, line: line)
        if result == .timedOut { _ = done.wait(timeout: .now() + 60) }
    }

    private func busyMessage(_ error: Error?) -> String? {
        guard case TranscribeError.busy(let message)? = error else { return nil }
        return message
    }

    private func isInvalidArgument(_ error: Error?) -> Bool {
        guard case TranscribeError.invalidArgument? = error else { return false }
        return true
    }

    // MARK: - Model-wide compute exclusion

    func testRunAndRunBatchWaitOnTheModelLock() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()

        let run = Box<Transcript>()
        assertWaitsOnModelLock(model, "run") { run.value = try? s1.run(pcm) }
        XCTAssertTrue(run.value?.text.lowercased().contains("country") == true)

        // A sibling session of the same model waits on the same lock.
        let batch = Box<[Result<Transcript, Error>]>()
        assertWaitsOnModelLock(model, "runBatch") { batch.value = try? s2.runBatch([pcm]) }
        XCTAssertEqual(batch.value?.count, 1)
        XCTAssertTrue(
            (try? batch.value?.first?.get().text.lowercased().contains("country")) == true)
    }

    func testEveryStreamSiteWaitsOnTheModelLock() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let session = try model.session()
        let chunk = Array(pcm.prefix(1600))
        let stream = Box<TranscribeCpp.Stream>()

        assertWaitsOnModelLock(model, "stream begin") { stream.value = try? session.stream() }
        XCTAssertNotNil(stream.value)
        XCTAssertTrue(model.streamActive)

        let fed = Box<StreamUpdate>()
        assertWaitsOnModelLock(model, "feed") { fed.value = try? stream.value?.feed(chunk) }
        XCTAssertNotNil(fed.value)

        let finalized = Box<StreamUpdate>()
        assertWaitsOnModelLock(model, "finalize") {
            finalized.value = try? stream.value?.finalize()
        }
        XCTAssertEqual(finalized.value?.isFinal, true)
        XCTAssertFalse(model.streamActive)
        stream.value = nil

        stream.value = try session.stream()
        _ = try stream.value?.feed(chunk)
        let state = Box<StreamState>()
        assertWaitsOnModelLock(model, "reset") { state.value = stream.value?.reset() }
        XCTAssertEqual(state.value, .idle)
        XCTAssertFalse(model.streamActive)
        stream.value = nil

        // A dropped ACTIVE stream resets under the lock in deinit. The last
        // reference is released on the background thread so the test thread
        // (which holds the lock) never runs the deinit itself.
        stream.value = try session.stream()
        _ = try stream.value?.feed(chunk)
        XCTAssertTrue(model.streamActive)
        assertWaitsOnModelLock(model, "Stream deinit") { stream.value = nil }
        XCTAssertFalse(model.streamActive)
        XCTAssertEqual(try session.stream().reset(), .idle)
    }

    // MARK: - Active-stream rejection and its ordering

    /// The `.busy` check runs only once the lock is held: with a stream active
    /// and the lock held elsewhere, run / runBatch / stream block first and
    /// then refuse. Also pins the exact busy messages.
    func testActiveStreamIsRefusedAfterTakingTheLock() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()
        let active = try s1.stream()
        _ = try active.feed(Array(pcm.prefix(1600)))

        let runError = Box<Error>()
        assertWaitsOnModelLock(model, "run while streaming") {
            do { _ = try s2.run(pcm) } catch { runError.value = error }
        }
        XCTAssertEqual(
            busyMessage(runError.value),
            "a stream is active on this model; finish or drop it before run()")

        let batchError = Box<Error>()
        assertWaitsOnModelLock(model, "runBatch while streaming") {
            do { _ = try s2.runBatch([pcm]) } catch { batchError.value = error }
        }
        XCTAssertEqual(
            busyMessage(batchError.value),
            "a stream is active on this model; finish or drop it before runBatch()")

        let streamError = Box<Error>()
        assertWaitsOnModelLock(model, "stream while streaming") {
            do { _ = try s2.stream() } catch { streamError.value = error }
        }
        XCTAssertEqual(busyMessage(streamError.value), "a stream is already active on this model")

        // The same session's second stream gets the binding's `.busy` too, not
        // a native error.
        XCTAssertEqual(
            busyMessage(Result { try s1.stream() }.failure),
            "a stream is already active on this model")

        active.reset()
    }

    /// String-option validation runs before the lock and before the busy
    /// check: with a stream active AND the lock held, a NUL in an option is
    /// reported as `.invalidArgument` without waiting.
    func testOptionValidationPrecedesLockAndBusyCheck() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()
        let active = try s1.stream()
        let bad = RunOptions(language: "e\0n")

        let runError = Box<Error>()
        assertSkipsModelLock(model, "run option check") {
            do { _ = try s2.run(pcm, options: bad) } catch { runError.value = error }
        }
        XCTAssertTrue(isInvalidArgument(runError.value), "\(String(describing: runError.value))")

        let batchError = Box<Error>()
        assertSkipsModelLock(model, "runBatch option check") {
            do { _ = try s2.runBatch([pcm], options: bad) } catch { batchError.value = error }
        }
        XCTAssertTrue(isInvalidArgument(batchError.value), "\(String(describing: batchError.value))")

        let streamError = Box<Error>()
        assertSkipsModelLock(model, "stream option check") {
            do { _ = try s2.stream(bad) } catch { streamError.value = error }
        }
        XCTAssertTrue(
            isInvalidArgument(streamError.value), "\(String(describing: streamError.value))")

        active.reset()
    }

    // MARK: - Cancel hook install / cleanup

    /// The async bridge installs a token only for the call and removes it
    /// afterwards, on success and after an abort alike.
    func testAsyncBridgeRemovesItsTokenAfterTheCall() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        XCTAssertNil(session.cancelToken)

        _ = try await session.run(pcm)
        XCTAssertNil(session.cancelToken, "bridged token must be removed after run")
        _ = try await session.runBatch([pcm])
        XCTAssertNil(session.cancelToken, "bridged token must be removed after runBatch")

        let long = Array(repeating: pcm, count: 6).flatMap { $0 }
        let task = Task { try await session.run(long) }
        task.cancel()
        _ = try? await task.value
        XCTAssertTrue(session.wasAborted)
        XCTAssertNil(session.cancelToken, "bridged token must be removed after an aborted run")

        // The cancelled token is not left behind: the next call is not aborted.
        _ = try await session.run(pcm)
        XCTAssertFalse(session.wasAborted)
        XCTAssertNil(session.cancelToken)
    }

    /// The synchronous path never installs or clears a token: a caller token
    /// stays installed (and stays cancelled) across runs until the caller
    /// clears it.
    func testSyncRunLeavesCallerTokenInstalled() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        let token = CancellationToken()
        token.cancel()
        session.setCancellationToken(token)
        for _ in 0..<2 {
            XCTAssertThrowsError(try session.run(pcm)) { error in
                guard case TranscribeError.aborted = error else {
                    return XCTFail("expected .aborted, got \(error)")
                }
            }
            XCTAssertTrue(session.cancelToken === token)
        }
        session.clearCancellationToken()
        XCTAssertNil(session.cancelToken)
        _ = try session.run(pcm)
        XCTAssertFalse(session.wasAborted)
    }

    // MARK: - Keep-alive

    func testSessionKeepsModelAliveForRun() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        weak var weakModel: Model?
        var session: Session? = try {
            let model = try Model(path: path)
            weakModel = model
            return try model.session()
        }()
        XCTAssertNotNil(weakModel, "a live Session must keep its Model alive")
        XCTAssertTrue(try session!.run(pcm).text.lowercased().contains("country"))
        session = nil
        XCTAssertNil(weakModel, "the Model is freed once its last Session is")
    }

    func testStreamKeepsSessionAndModelAlive() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        weak var weakModel: Model?
        weak var weakSession: Session?
        var stream: TranscribeCpp.Stream? = try {
            let model = try Model(path: path)
            let session = try model.session()
            weakModel = model
            weakSession = session
            return try session.stream()
        }()
        XCTAssertNotNil(weakSession, "a live Stream must keep its Session alive")
        XCTAssertNotNil(weakModel, "a live Stream must keep its Model alive")
        try Fixtures.drive(stream!, pcm: pcm)
        XCTAssertTrue(stream!.text.full.lowercased().contains("country"))
        stream = nil
        XCTAssertNil(weakSession)
        XCTAssertNil(weakModel)
    }

    /// The caller may drop every reference while an async call is in flight:
    /// the call itself keeps the session (and so the model) alive until it
    /// returns.
    func testCallerDropsReferencesDuringAsyncRun() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let long = Array(repeating: pcm, count: 3).flatMap { $0 }
        let task: Task<Transcript, Error> = try {
            let session = try Model(path: path).session()
            return Task { try await session.run(long) }
        }()
        let transcript = try await task.value
        XCTAssertTrue(transcript.text.lowercased().contains("country"))
    }

    // MARK: - Copy-out

    /// Results are owned copies: later compute on the same session or on a
    /// sibling session leaves them untouched.
    func testResultsSurviveLaterCompute() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()
        let first = try s1.run(pcm)
        let batch = try s1.runBatch([pcm])
        let text = first.text
        let segments = first.segments.count
        let batchText = try batch[0].get().text
        XCTAssertTrue(text.lowercased().contains("country"))

        let short = Array(pcm.prefix(16000))
        _ = try s1.run(short)
        _ = try s1.runBatch([short])
        _ = try s2.run(short)

        XCTAssertEqual(first.text, text)
        XCTAssertEqual(first.segments.count, segments)
        XCTAssertEqual(try batch[0].get().text, batchText)
        XCTAssertNotEqual(try s1.run(short).text, text)
    }

    func testStreamSnapshotSurvivesReset() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream()
        try Fixtures.drive(stream, pcm: pcm)
        let snapshot = stream.snapshot
        let text = stream.text
        XCTAssertTrue(text.full.lowercased().contains("country"))
        stream.reset()
        XCTAssertEqual(stream.snapshot.text, "")
        XCTAssertTrue(snapshot.text.lowercased().contains("country"))
        XCTAssertTrue(text.full.lowercased().contains("country"))
    }
}

private extension Result {
    var failure: Failure? {
        if case .failure(let error) = self { return error }
        return nil
    }
}
