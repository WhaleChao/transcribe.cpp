import CTranscribe
import XCTest

@testable import TranscribeCpp

/// DIARIZE role (`DiarizeSession`). Lock / lease / keep-alive rules are in
/// ComputeRulesTests.
final class DiarizeTests: XCTestCase {
    /// `p` is NaN on diarize rows, so compare the turns, not `SpeakerSegment`s.
    private func turns(_ rows: [SpeakerSegment]) -> [[Int64]] {
        rows.map { [$0.t0Ms, $0.t1Ms, Int64($0.speakerId)] }
    }

    private func assertUnsupportedRole(_ calls: [(String, () throws -> Void)]) {
        for (what, call) in calls {
            XCTAssertThrowsError(try call(), what) { error in
                guard case TranscribeError.unsupportedRole = error else {
                    return XCTFail("\(what): expected .unsupportedRole, got \(error)")
                }
            }
        }
    }

    func testRoleBitsMatchTheHeader() {
        XCTAssertEqual(Roles.asr.rawValue, 1)
        XCTAssertEqual(Roles.diarize.rawValue, 2)
    }

    func testAsrOnlyModelRefusesDiarize() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        XCTAssertEqual(model.roles, .asr)
        XCTAssertFalse(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        assertUnsupportedRole([
            ("diarizeInfo", { _ = try model.diarizeInfo }),
            ("diarizeSession", { _ = try model.diarizeSession() }),
        ])
    }

    func testSortformerDiarizes() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        // The golden turns are CPU numerics.
        let model = try Model(path: path, options: ModelOptions(backend: .cpu))
        XCTAssertEqual(model.roles, [.diarize])
        XCTAssertEqual(try model.diarizeInfo, DiarizeInfo(sampleRate: 16000, maxSpeakers: 4))
        XCTAssertTrue(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        assertUnsupportedRole([
            ("capabilities", { _ = try model.capabilities }),
            ("session", { _ = try model.session() }),
        ])

        // (t0Ms, t1Ms, speakerId) on samples/sortformer-2spk-mix.wav. Only
        // .lowLatency moves the turns here, so it shows the preset reaches the
        // native run.
        let golden: [(SortformerPreset, [[Int64]])] = [
            (.default, [[320, 2400, 1], [7360, 9360, 1], [10240, 10640, 1],
                        [4240, 6640, 2], [9760, 12000, 2]]),
            (.lowLatency, [[320, 2480, 1], [7360, 9360, 1], [10240, 10640, 1],
                           [4160, 6640, 2], [9760, 12000, 2]]),
        ]
        let session = try model.diarizeSession()
        for (preset, want) in golden {
            let rows = try session.run(
                pcm, options: DiarizeOptions(family: .sortformer(SortformerDiarizeOptions(preset: preset))))
            XCTAssertEqual(turns(rows), want, "\(preset)")
        }
        XCTAssertGreaterThan(session.timings.encodeMs, 0)
        XCTAssertEqual(turns(try session.run(pcm)), golden[0].1, "no extension == .default")
    }

    func testBadPresetAndCancellation() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        let session = try Model(path: path).diarizeSession()
        let first = try session.run(pcm)
        XCTAssertFalse(first.isEmpty)

        // `SortformerPreset` cannot express an out-of-range preset, so the
        // native rejection is driven through the raw C call.
        var ext = transcribe_sortformer_diarize_ext()
        transcribe_sortformer_diarize_ext_init(&ext)
        ext.preset = transcribe_sortformer_preset(rawValue: 99)
        let status = withUnsafePointer(to: &ext.ext) { family in
            var params = transcribe_diarize_params()
            transcribe_diarize_params_init(&params)
            params.family = family
            return pcm.withUnsafeBufferPointer {
                transcribe_diarize_run(session.ptr, $0.baseAddress, Int32($0.count), &params)
            }
        }
        guard case .invalidArgument = TranscribeError.make(status) else {
            return XCTFail("expected .invalidArgument, got \(status)")
        }
        XCTAssertEqual(Int(transcribe_diarize_n_segments(session.ptr)), first.count)

        let token = CancellationToken()
        token.cancel()
        session.setCancellationToken(token)
        XCTAssertThrowsError(try session.run(pcm)) { error in
            guard case TranscribeError.aborted = error else {
                return XCTFail("expected .aborted, got \(error)")
            }
        }
        session.clearCancellationToken()
        XCTAssertEqual(turns(try session.run(pcm)), turns(first))
    }
}
