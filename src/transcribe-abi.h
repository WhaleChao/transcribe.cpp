// transcribe-abi.h - internal helpers for the size-aware public ABI.
//
// Shared by the public dispatchers (transcribe.cpp, transcribe-asr.cpp) and
// per-family public accessors (e.g. arch/whisper/public.cpp) so the raw enum
// reads, struct_size validation and copy-out truncation logic live in
// exactly one place. Not part of the public API.

#pragma once

#include "transcribe.h"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace transcribe {

// Raw enum reads at the public ABI boundary
//
// C callers can store ANY int in an enum-typed ABI field; in C++ loading an
// out-of-range value through an enum-typed lvalue is UB (UBSan traps it). So
// every public entry point reads caller-supplied enum values as raw bytes
// first, validates the raw int, and only then converts to the enum type. All
// public enums are int-sized; the static_assert below pins one.

inline int enum_field_raw(const void * field) {
    int raw;
    std::memcpy(&raw, field, sizeof(raw));
    return raw;
}

static_assert(sizeof(transcribe_backend_request) == sizeof(int),
              "public enums must be int-sized for raw boundary reads");

// Every PCM entry point rejects non-finite samples (NaN / +-Inf) with
// TRANSCRIBE_ERR_INVALID_ARG before it modifies any result or stream state:
// a NaN would otherwise flow through the front end and come out as a
// confident-looking result. Silence is valid input. O(n), negligible next to
// any encoder.
inline bool pcm_is_finite(const float * pcm, int n_samples) {
    for (int i = 0; i < n_samples; ++i) {
        if (!std::isfinite(pcm[i])) {
            return false;
        }
    }
    return true;
}

// Strict size check for every caller-owned struct (inputs AND outputs):
// struct_size smaller than the prefix the library relies on is rejected.
// `0` is invalid — defaults are reached by passing NULL where the entry
// point accepts a nullable params pointer, never by a zero-sized struct.
inline transcribe_status check_struct_size(uint64_t got, uint64_t want) {
    if (got < want) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    return TRANSCRIBE_OK;
}

// Alias for call-site readability ("this is an input params struct").
// Behavior is identical to check_struct_size: struct_size == 0 is rejected
// (so coincidentally-zero stack memory is not taken as a defaults
// shortcut); defaults come from NULL.
inline transcribe_status check_input_struct_size(uint64_t got, uint64_t want) {
    return check_struct_size(got, want);
}

// Copy min(caller_size, library_size) bytes from a library-staged struct
// into the caller's output buffer. The min() is the overflow guard: the
// library never writes past what the caller declared, and never reads
// past what it staged. caller_size is uint64_t (matches the public
// struct_size field type); library_size is size_t (always sizeof(T)).
inline void copy_out_prefix(void * dst, const void * src, uint64_t caller_size, size_t library_size) {
    const uint64_t lib = static_cast<uint64_t>(library_size);
    const uint64_t n   = caller_size < lib ? caller_size : lib;
    std::memcpy(dst, src, static_cast<size_t>(n));
}

}  // namespace transcribe
