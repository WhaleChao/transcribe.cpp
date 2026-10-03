import Foundation
import XCTest

@testable import TranscribeCpp

/// Every compute site waits on the model-wide lock; option checks run before it, the busy check after.
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

    /// A diarize run follows the Session rules: it keeps its model alive,
    /// waits on the model lock, and refuses a held stream lease with `.busy`
    /// only once the lock is taken.
    func testDiarizeRunSharesTheModelLockAndLease() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        weak var weakModel: Model?
        let session: DiarizeSession = try {
            let model = try Model(path: path)
            weakModel = model
            return try model.diarizeSession()
        }()
        XCTAssertNotNil(weakModel, "a live DiarizeSession must keep its Model alive")
        let model = session.model

        let rows = Box<[SpeakerSegment]>()
        assertWaitsOnModelLock(model, "diarize run") { rows.value = try? session.run(pcm) }
        XCTAssertEqual(rows.value?.isEmpty, false)

        // Sortformer cannot stream, so the lease a stream would hold is set directly.
        model.withCompute { model.streamActive = true }
        defer { model.withCompute { model.streamActive = false } }
        let error = Box<Error>()
        assertWaitsOnModelLock(model, "diarize run while streaming") {
            do { _ = try session.run(pcm) } catch let e { error.value = e }
        }
        XCTAssertEqual(
            busyMessage(error.value),
            "a stream is active on this model; finish or drop it before diarize run()")
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

    // MARK: - Keep-alive

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
}
