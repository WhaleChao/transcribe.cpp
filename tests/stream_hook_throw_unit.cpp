// stream_hook_throw_unit.cpp - a family stream hook that throws ends the
// stream: begin / feed / finalize leave it FAILED with last_status matching
// the status the call returned (OOM for bad_alloc, BACKEND otherwise), and
// reset still ends IDLE. Before, a throwing hook left the stream ACTIVE
// while the caller got an error.

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

enum class Throw { None, BadAlloc, Runtime };

Throw g_begin    = Throw::None;
Throw g_feed     = Throw::None;
Throw g_finalize = Throw::None;
Throw g_reset    = Throw::None;

void maybe_throw(Throw t) {
    if (t == Throw::BadAlloc) {
        throw std::bad_alloc();
    }
    if (t == Throw::Runtime) {
        throw std::runtime_error("hook failed");
    }
}

transcribe_status fake_begin(transcribe_session *, const transcribe_run_params *, const transcribe_stream_params *) {
    maybe_throw(g_begin);
    return TRANSCRIBE_OK;
}

transcribe_status fake_feed(transcribe_session *, const float *, int, transcribe_stream_update *) {
    maybe_throw(g_feed);
    return TRANSCRIBE_OK;
}

transcribe_status fake_finalize(transcribe_session *, transcribe_stream_update *) {
    maybe_throw(g_finalize);
    return TRANSCRIBE_OK;
}

void fake_reset(transcribe_session *) {
    maybe_throw(g_reset);
}

const transcribe::Arch k_arch = {
    /* .name             = */ "fake-stream-throw",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ fake_begin,
    /* .stream_feed      = */ fake_feed,
    /* .stream_finalize  = */ fake_finalize,
    /* .stream_reset     = */ fake_reset,
};

void reset_globals() {
    g_begin = g_feed = g_finalize = g_reset = Throw::None;
}

struct Fixture {
    transcribe_model   model;
    transcribe_session session;
    std::vector<float> pcm = std::vector<float>(1600, 0.0f);

    Fixture() {
        model.arch                    = &k_arch;
        model.caps.supports_streaming = true;
        session.model                 = &model;
    }

    transcribe_status feed() {
        return transcribe_stream_feed(&session, pcm.data(), static_cast<int>(pcm.size()), nullptr);
    }
};

void expect_failed(const transcribe_session & s, transcribe_status st, transcribe_status want) {
    CHECK(st == want);
    CHECK(transcribe_stream_get_state(&s) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(&s) == want);
}

void test_begin_throw() {
    for (Throw t : { Throw::BadAlloc, Throw::Runtime }) {
        reset_globals();
        Fixture f;
        g_begin                    = t;
        const transcribe_status st = transcribe_stream_begin(&f.session, nullptr, nullptr);
        expect_failed(f.session, st, t == Throw::BadAlloc ? TRANSCRIBE_ERR_OOM : TRANSCRIBE_ERR_BACKEND);
        // A failed stream accepts a fresh begin.
        g_begin = Throw::None;
        CHECK(transcribe_stream_begin(&f.session, nullptr, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_stream_get_state(&f.session) == TRANSCRIBE_STREAM_ACTIVE);
    }
}

void test_feed_throw() {
    for (Throw t : { Throw::BadAlloc, Throw::Runtime }) {
        reset_globals();
        Fixture f;
        CHECK(transcribe_stream_begin(&f.session, nullptr, nullptr) == TRANSCRIBE_OK);
        g_feed = t;
        expect_failed(f.session, f.feed(), t == Throw::BadAlloc ? TRANSCRIBE_ERR_OOM : TRANSCRIBE_ERR_BACKEND);
        // No longer ACTIVE, so further feeds are refused as on any failed stream.
        g_feed = Throw::None;
        CHECK(f.feed() == TRANSCRIBE_ERR_INVALID_ARG);
    }
}

void test_finalize_throw() {
    for (Throw t : { Throw::BadAlloc, Throw::Runtime }) {
        reset_globals();
        Fixture f;
        CHECK(transcribe_stream_begin(&f.session, nullptr, nullptr) == TRANSCRIBE_OK);
        CHECK(f.feed() == TRANSCRIBE_OK);
        g_finalize                 = t;
        const transcribe_status st = transcribe_stream_finalize(&f.session, nullptr);
        expect_failed(f.session, st, t == Throw::BadAlloc ? TRANSCRIBE_ERR_OOM : TRANSCRIBE_ERR_BACKEND);
    }
}

void test_reset_throw_still_idles() {
    reset_globals();
    Fixture f;
    CHECK(transcribe_stream_begin(&f.session, nullptr, nullptr) == TRANSCRIBE_OK);
    f.session.full_text   = "partial";
    f.session.has_result  = true;
    f.session.was_aborted = true;
    g_reset               = Throw::Runtime;
    transcribe_stream_reset(&f.session);  // void; the guard swallows the exception
    CHECK(transcribe_stream_get_state(&f.session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(!f.session.has_result);
    CHECK(f.session.full_text.empty());
    CHECK(!transcribe_was_aborted(&f.session));
    g_reset = Throw::None;
    CHECK(transcribe_stream_begin(&f.session, nullptr, nullptr) == TRANSCRIBE_OK);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);  // the guard logs each caught exception

    test_begin_throw();
    test_feed_throw();
    test_finalize_throw();
    test_reset_throw_still_idles();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("stream_hook_throw_unit: ok\n");
    return EXIT_SUCCESS;
}
