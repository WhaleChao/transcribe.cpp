// role_resolve_unit.cpp - load-time role validation (transcribe::resolve_roles).

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>

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

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("role_resolve_unit: ok\n");
    return EXIT_SUCCESS;
}
