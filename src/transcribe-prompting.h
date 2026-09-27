// transcribe-prompting.h - shared helpers for the generic prompting fields
// (transcribe_run_params::vocabulary / prompt / prefix, TASK_INSTRUCT).
//
// INTERNAL. The dispatcher validates the fields and removes the ones a model
// ignores before a family sees them, so a family acts on whatever is set in
// the params view it receives. Families own the rendering (where the text
// goes in their prompt) and their budget rules; these helpers cover the parts
// every family shares.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe {

class Tokenizer;

namespace prompting {

inline bool has_text(const char * s) {
    return s != nullptr && s[0] != '\0';
}

inline bool is_instruct(const transcribe_run_params * p) {
    return p != nullptr && p->task == TRANSCRIBE_TASK_INSTRUCT;
}

// Non-empty vocabulary terms in caller order. Assumes the dispatcher's
// shape validation (non-negative count, no NULL entries) already passed.
std::vector<std::string> terms(const transcribe_run_params * p);

std::string join(const std::vector<std::string> & terms, const char * sep);

// Rejects user text containing a literal of one of the tokenizer's control
// tokens (e.g. "<|im_end|>", "[INST]"). The encoders never produce control
// ids from text, but the upstream reference tokenizers do, so accepting such
// text would silently diverge from the reference and could close a chat
// turn. `what` names the field in the error log. Returns INVALID_ARG on a hit.
transcribe_status check_plain_text(const Tokenizer & tok, const std::string & text, const char * what);

// check_plain_text + encode.
transcribe_status encode_plain(const Tokenizer &      tok,
                               const std::string &    text,
                               std::vector<int32_t> & out_ids,
                               const char *           what);

// Rendered-prompt observability for parity tests. Decodes `ids` with special
// pieces kept, collapsing each run of `audio_id` to "<piece>xN" (the reference
// harness format). When TRANSCRIBE_PROMPT_DUMP names a file the line is
// appended there; otherwise it is logged at DEBUG (truncated to the log
// line limit).
void dump_rendered(const Tokenizer & tok, const std::vector<int32_t> & ids, int32_t audio_id, const char * family);

}  // namespace prompting
}  // namespace transcribe
