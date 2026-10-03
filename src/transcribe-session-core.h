// transcribe-session-core.h - internal base shared by every role session.
//
// SessionCore holds what is not specific to any role: the borrowed model,
// the thread count, the per-run ggml compute scratch, the abort callback and
// the stage timings. transcribe_session (ASR) derives from it, and so does
// each role's session type, so scratch release (Handy #2000), cancellation
// and timings behave the same for every role. Families keep reaching these
// members through their derived context (ctx->sched, ctx->poll_abort(), ...).

#pragma once

#include "transcribe.h"

#include <cmath>
#include <cstdint>

struct transcribe_model;

// ggml handle types for the base-owned compute scratch. Forward-declared so
// this internal header does not pull the ggml headers into every includer
// (the public transcribe.h never sees them); transcribe-backend.h uses the
// same pattern for ggml_backend_t.
struct ggml_context;
struct ggml_backend_sched;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

namespace transcribe {

// "Who spoke when" rows (diarization). Populated only when a run
// resolves diarize ON for a supporting family; may overlap in time.
// t0_ms == t1_ms == 0 means the family attributes text but has no
// timing information for the turn.
struct SpeakerSegmentEntry {
    int64_t t0_ms      = 0;
    int64_t t1_ms      = 0;
    int32_t speaker_id = 0;  // 1-based
    float   p          = NAN;
};

struct SessionCore {
    // The model this session was constructed from. Borrowed pointer:
    // the caller is required (per the public threading contract) to keep
    // the model alive for the lifetime of every derived session.
    transcribe_model * model = nullptr;

    // Cached n_threads value the caller passed at init time. 0 means
    // "library picks a sensible default" (matches the factory).
    int n_threads = 0;

    // Per-call timings, populated by the most recent transcribe_run.
    // Surfaced via the public transcribe_get_timings accessor; reset
    // by transcribe_reset_timings.
    int64_t t_mel_us    = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;

    // Abort / cancellation (set via transcribe_set_abort_callback).
    // run() drivers call poll_abort() at chunk / decode-step boundaries;
    // a callback returning true sets was_aborted and the run returns
    // TRANSCRIBE_ERR_ABORTED with partial segments preserved. was_aborted
    // is cleared at the top of every transcribe_run, NOT by clear_result
    // (the partial result may be deliberately retained).
    transcribe_abort_callback abort_cb       = nullptr;
    void *                    abort_userdata = nullptr;
    bool                      was_aborted    = false;

    bool poll_abort() {
        if (abort_cb != nullptr && abort_cb(abort_userdata)) {
            was_aborted = true;
            return true;
        }
        return false;
    }

    // Per-run ggml compute scratch, owned by the base so every family
    // releases it the same way and none can forget to: the backend
    // scheduler (whose graph allocator only ever grows) and the no_alloc
    // graph context. Families create both lazily inside their run / stream
    // hooks and use them directly; the base frees them in release_scratch
    // and in its destructor (scheduler first, then context).
    ggml_backend_sched_t sched       = nullptr;
    ggml_context *       compute_ctx = nullptr;

    // Release the per-run ggml compute scratch (sched, then compute_ctx;
    // see transcribe::release_compute_scratch), then let the family drop
    // pointers that lived in them (on_scratch_released).
    // The dispatcher calls this after every offline transcribe_run or
    // transcribe_run_batch that passes pre-clear validation and reaches its
    // commit point, whether family execution succeeds, fails, or throws.
    // Without this, a single long utterance would pin the scheduler's
    // high-water mark in backend compute memory for the session's lifetime
    // (Handy #2000). Families re-create the scheduler lazily on the next run.
    // The measured recreation cost is about 1 ms per run on Metal and up to
    // about 10 ms on CPU for families that reserve a worst-case decoder
    // workspace each run, such as Canary and Cohere.
    //
    // Host-side vectors that scale with input length, such as mel buffers,
    // encoder host copies, and positional banks, deliberately retain their
    // capacity. Family KV caches sized by the last batch are also retained;
    // this hook releases only the ggml scheduler and compute context.
    //
    // Streaming entry points do not invoke this hook. A streaming session
    // keeps its scheduler until a later offline run or session destruction.
    // Parakeet and Voxtral Realtime use per-chunk bounded stream workspaces;
    // Moonshine Streaming's decode graph cross-attends over the complete
    // committed stream, so its workspace grows with total stream length.
    // Must not throw. Non-virtual: a family cannot opt out of the release.
    void release_scratch() noexcept;

    SessionCore() = default;
    // Frees sched / compute_ctx after the derived destructors have run. The
    // scheduler owns only its own allocator buffers and references
    // model-owned backends, so freeing it after the family's KV caches and
    // stream buffers is order-independent.
    virtual ~SessionCore();

    SessionCore(const SessionCore &)             = delete;
    SessionCore & operator=(const SessionCore &) = delete;
    SessionCore(SessionCore &&)                  = delete;
    SessionCore & operator=(SessionCore &&)      = delete;

  protected:
    // Family hook, called by release_scratch after sched / compute_ctx are
    // freed: null out tensors borrowed from the freed context (encoder_out)
    // or reset capacity bookkeeping (whisper compute_ctx_size). Most
    // families need nothing. Not called from the base destructor because
    // derived members are already destroyed by then. Must not throw.
    virtual void on_scratch_released() noexcept {}
};

}  // namespace transcribe
