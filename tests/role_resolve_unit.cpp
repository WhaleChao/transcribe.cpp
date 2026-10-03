// role_resolve_unit.cpp - load-time role validation (transcribe::resolve_roles) and the public role API (transcribe_model_roles, UNSUPPORTED_ROLE).

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

transcribe_status fake_init_context(transcribe_model *, const transcribe_session_params *, transcribe_session **) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

transcribe_status fake_run(transcribe_session *, const float *, int, const transcribe_run_params *) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

const transcribe::Arch k_asr_arch = {
    /* .name             = */ "fake_asr",
    /* .load             = */ nullptr,
    /* .init_context     = */ fake_init_context,
    /* .run              = */ fake_run,
};

// init_context without run: not a usable ASR arch.
const transcribe::Arch k_half_asr_arch = {
    /* .name             = */ "fake_half_asr",
    /* .load             = */ nullptr,
    /* .init_context     = */ fake_init_context,
};

const transcribe::Arch k_no_role_arch = {
    /* .name             = */ "fake_no_role",
};

transcribe_status resolve(const transcribe::Arch & arch, uint32_t roles, uint32_t * out_roles) {
    transcribe_model model;
    model.arch                 = &arch;
    model.roles                = roles;
    const transcribe_status st = transcribe::resolve_roles(&model);
    *out_roles                 = model.roles;
    return st;
}

void test_asr_default() {
    uint32_t roles = 0;
    CHECK(resolve(k_asr_arch, 0, &roles) == TRANSCRIBE_OK);
    CHECK(roles == transcribe::k_role_asr);
}

void test_asr_explicit() {
    uint32_t roles = 0;
    CHECK(resolve(k_asr_arch, transcribe::k_role_asr, &roles) == TRANSCRIBE_OK);
    CHECK(roles == transcribe::k_role_asr);
}

void test_no_role_rejected() {
    uint32_t roles = 0;
    CHECK(resolve(k_no_role_arch, 0, &roles) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(resolve(k_half_asr_arch, 0, &roles) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

void test_unbacked_role_rejected() {
    uint32_t roles = 0;
    CHECK(resolve(k_half_asr_arch, transcribe::k_role_asr, &roles) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(resolve(k_asr_arch, transcribe::k_role_diarize, &roles) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(resolve(k_asr_arch, transcribe::k_role_asr | transcribe::k_role_diarize, &roles) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

void test_unknown_bit_rejected() {
    uint32_t roles = 0;
    CHECK(resolve(k_asr_arch, transcribe::k_role_asr | (1u << 31), &roles) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

// D5: ASR entry points refuse a model that does not serve ASR, before
// touching params or the arch; capabilities leave the caller's struct alone.
void test_asr_entry_points_check_role() {
    transcribe_model model;
    model.arch  = &k_asr_arch;
    model.roles = transcribe::k_role_diarize;  // not ASR

    transcribe_session * s = reinterpret_cast<transcribe_session *>(0x1);
    CHECK(transcribe_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    caps.max_audio_ms = 1234;  // sentinel: must survive the rejection
    CHECK(transcribe_model_get_capabilities(&model, &caps) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(caps.max_audio_ms == 1234);

    CHECK(transcribe_model_roles(&model) == TRANSCRIBE_ROLE_DIARIZE);

    // Same model with the ASR bit: capabilities succeed; session_init gets
    // past the role check to the fake arch's init_context.
    model.roles = transcribe::k_role_asr;
    CHECK(transcribe_model_get_capabilities(&model, &caps) == TRANSCRIBE_OK);
    CHECK(transcribe_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(s == nullptr);
    CHECK(transcribe_model_roles(&model) == TRANSCRIBE_ROLE_ASR);
}

void test_role_bits_match_public_enum() {
    static_assert(transcribe::k_role_asr == TRANSCRIBE_ROLE_ASR, "internal and public role bits must match");
    static_assert(transcribe::k_role_diarize == TRANSCRIBE_ROLE_DIARIZE, "internal and public role bits must match");
    CHECK(std::strcmp(transcribe_status_string(TRANSCRIBE_ERR_UNSUPPORTED_ROLE), "unknown status") != 0);
}

void test_null_model() {
    CHECK(transcribe::resolve_roles(nullptr) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    transcribe_model model;
    CHECK(transcribe::resolve_roles(&model) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);  // the rejections log at ERROR

    test_asr_default();
    test_asr_explicit();
    test_no_role_rejected();
    test_unbacked_role_rejected();
    test_unknown_bit_rejected();
    test_null_model();
    test_asr_entry_points_check_role();
    test_role_bits_match_public_enum();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("role_resolve_unit: ok\n");
    return EXIT_SUCCESS;
}
