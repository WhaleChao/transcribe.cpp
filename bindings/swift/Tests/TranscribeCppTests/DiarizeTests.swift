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

    func testRoleBitsMatchTheHeader() {
        XCTAssertEqual(Roles.asr.rawValue, 1)
        XCTAssertEqual(Roles.diarize.rawValue, 2)
    }

    func testAsrOnlyModelRefusesDiarize() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        XCTAssertEqual(model.roles, .asr)
        XCTAssertFalse(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        for (what, call) in [
            ("diarizeInfo", { _ = try model.diarizeInfo }),
            ("diarizeSession", { _ = try model.diarizeSession() }),
        ] as [(String, () throws -> Void)] {
            XCTAssertThrowsError(try call(), what) { error in
                guard case TranscribeError.unsupportedRole = error else {
                    return XCTFail("\(what): expected .unsupportedRole, got \(error)")
                }
            }
        }
    }

    func testSortformerDiarizes() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        let model = try Model(path: path)
        XCTAssertEqual(model.roles, [.asr, .diarize])
        XCTAssertEqual(try model.diarizeInfo, DiarizeInfo(sampleRate: 16000, maxSpeakers: 4))
        XCTAssertTrue(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        XCTAssertGreaterThan(try model.capabilities.nativeSampleRate, 0)

        let session = try model.diarizeSession()
        // On this clip only .lowLatency moves the turns, so it shows the preset
        // reaches the native run.
        let preset = SortformerPreset.lowLatency
        let rows = try session.run(
            pcm, options: DiarizeOptions(family: .sortformer(SortformerDiarizeOptions(preset: preset))))
        XCTAssertFalse(rows.isEmpty)
        XCTAssertTrue(rows.allSatisfy { (1...4).contains($0.speakerId) && $0.t0Ms < $0.t1Ms })
        XCTAssertGreaterThan(session.timings.encodeMs, 0)
        XCTAssertNotEqual(turns(rows), turns(try session.run(pcm)))

        // TEMPORARY: the Sortformer ASR path (SFST on the RUN slot) is removed
        // in a later commit; until then both roles must yield the same turns.
        let asr = try model.session().run(
            pcm, options: RunOptions(family: .sortformer(SortformerStreamOptions(preset: preset))))
        XCTAssertEqual(turns(rows), turns(asr.speakerSegments))
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
