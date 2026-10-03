// diarize.cpp - transcribe-cli DIARIZE driver: speaker segments for one file
// through include/transcribe/diarize.h.

#include "transcribe/diarize.h"

#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int transcribe_cli::run_diarize_file(const cli_args &           args,
                                     transcribe_model *         model,
                                     const std::vector<float> & pcm,
                                     double                     duration_s) {
    transcribe_diarize_session_params sp;
    transcribe_diarize_session_params_init(&sp);
    sp.n_threads                         = args.n_threads;
    transcribe_diarize_session * session = nullptr;
    transcribe_status            st      = transcribe_diarize_session_init(model, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "diarize session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    st = transcribe_diarize_run(session, pcm.data(), static_cast<int>(pcm.size()), nullptr);
    std::printf("run: %s\n", transcribe_status_string(st));
    if (st == TRANSCRIBE_OK) {
        const int n = transcribe_diarize_n_segments(session);
        std::printf("speaker segments: %d\n", n);
        for (int i = 0; i < n; ++i) {
            transcribe_speaker_segment row;
            transcribe_speaker_segment_init(&row);
            transcribe_diarize_get_segment(session, i, &row);
            std::printf("  [%7.2f -> %7.2f] S%d\n", row.t0_ms / 1000.0, row.t1_ms / 1000.0, row.speaker_id);
        }
    }

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    transcribe_diarize_get_timings(session, &tm);
    const double total_ms = tm.mel_ms + tm.encode_ms;
    if (total_ms > 0.0 && duration_s > 0.0) {
        std::printf("  realtime:   %.0fx (%.1f ms for %.1f s)\n", duration_s * 1000.0 / total_ms, total_ms, duration_s);
    }

    transcribe_diarize_session_free(session);
    transcribe_model_free(model);
    return st == TRANSCRIBE_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
