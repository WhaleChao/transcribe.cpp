// transcribe-prompting.cpp - shared helpers for the generic prompting fields.

#include "transcribe-prompting.h"

#include "transcribe-env.h"
#include "transcribe-log.h"
#include "transcribe-tokenizer.h"

#include <cstdio>

namespace transcribe::prompting {

std::vector<std::string> terms(const transcribe_run_params * p) {
    std::vector<std::string> out;
    if (p == nullptr || p->vocabulary == nullptr || p->n_vocabulary <= 0) {
        return out;
    }
    out.reserve(static_cast<size_t>(p->n_vocabulary));
    for (int32_t i = 0; i < p->n_vocabulary; ++i) {
        if (has_text(p->vocabulary[i])) {
            out.emplace_back(p->vocabulary[i]);
        }
    }
    return out;
}

std::string join(const std::vector<std::string> & terms, const char * sep) {
    std::string out;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i != 0) {
            out += sep;
        }
        out += terms[i];
    }
    return out;
}

transcribe_status check_plain_text(const Tokenizer & tok, const std::string & text, const char * what) {
    // Candidate literals are "<...>" and "[...]" spans up to a control
    // token's plausible length. "<|...|>" pieces are rejected whenever the
    // vocab has them (Whisper's rule: those are never plain text); other
    // shapes only when the vocab types them CONTROL or they are BOS/EOS.
    constexpr size_t k_max_literal = 48;
    for (size_t i = 0; i < text.size(); ++i) {
        const char open = text[i];
        if (open != '<' && open != '[') {
            continue;
        }
        const char   close = open == '<' ? '>' : ']';
        const size_t end   = text.find(close, i + 1);
        if (end == std::string::npos || end - i + 1 > k_max_literal) {
            continue;
        }
        const std::string piece = text.substr(i, end - i + 1);
        const int         id    = tok.find(piece);
        if (id < 0) {
            continue;
        }
        const bool pipe_form = piece.size() >= 4 && piece[1] == '|' && piece[piece.size() - 2] == '|';
        if (pipe_form || tok.is_control(id) || id == tok.bos_id() || id == tok.eos_id()) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s contains the control token \"%s\" (id %d); control tokens are "
                    "not accepted in prompting text", what, piece.c_str(), id);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status encode_plain(const Tokenizer &      tok,
                               const std::string &    text,
                               std::vector<int32_t> & out_ids,
                               const char *           what) {
    if (const transcribe_status st = check_plain_text(tok, text, what); st != TRANSCRIBE_OK) {
        return st;
    }
    out_ids.clear();
    if (text.empty()) {
        return TRANSCRIBE_OK;
    }
    return tok.encode(text, out_ids);
}

void dump_rendered(const Tokenizer & tok, const std::vector<int32_t> & ids, int32_t audio_id, const char * family) {
    std::string out;
    for (size_t i = 0; i < ids.size();) {
        const int id = ids[i];
        size_t    j  = i + 1;
        if (id == audio_id) {
            while (j < ids.size() && ids[j] == id) {
                ++j;
            }
        }
        out += tok.decode(&id, 1);
        if (j - i > 1) {
            out += "x" + std::to_string(j - i);
        }
        i = j;
    }
    if (const char * path = env::str("TRANSCRIBE_PROMPT_DUMP")) {
        if (std::FILE * f = std::fopen(path, "ab")) {
            std::fprintf(f, "%s\t%zu\t", family, ids.size());
            std::fwrite(out.data(), 1, out.size(), f);
            std::fputc('\n', f);
            std::fclose(f);
        }
        return;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "%s prompt (%zu tokens): %s", family, ids.size(), out.c_str());
}

}  // namespace transcribe::prompting
