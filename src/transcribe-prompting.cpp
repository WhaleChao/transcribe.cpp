// transcribe-prompting.cpp - shared helpers for the generic prompting fields.

#include "transcribe-prompting.h"

#include "transcribe-env.h"
#include "transcribe-log.h"
#include "transcribe-tokenizer.h"

#include <algorithm>
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "%s contains the control token \"%s\" (id %d); control tokens are "
                    "not accepted in prompting text",
                    what, piece.c_str(), id);
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

transcribe_status fit_terms_and_context(const Tokenizer &                tok,
                                        const std::vector<std::string> & terms_in,
                                        const TermsFormat &              fmt,
                                        const std::string &              ctx,
                                        int                              budget,
                                        const char *                     family,
                                        FittedPrompt &                   out) {
    out                                   = FittedPrompt{};
    std::vector<std::string> terms        = terms_in;
    auto                     encode_terms = [&]() -> transcribe_status {
        out.term_ids.clear();
        return terms.empty() ?
                   TRANSCRIBE_OK :
                   encode_plain(tok, fmt.lead + join(terms, fmt.sep.c_str()) + fmt.trail, out.term_ids, "vocabulary");
    };
    if (const transcribe_status st = encode_terms(); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = encode_plain(tok, ctx, out.ctx_ids, "prompt"); st != TRANSCRIBE_OK) {
        return st;
    }
    const size_t ctx_in = out.ctx_ids.size();
    if (budget >= 0 && out.term_ids.size() + out.ctx_ids.size() > static_cast<size_t>(budget)) {
        const size_t cap  = static_cast<size_t>(budget);
        const size_t room = out.term_ids.size() < cap ? cap - out.term_ids.size() : 0;
        out.ctx_ids.erase(out.ctx_ids.begin(), out.ctx_ids.end() - std::min(room, out.ctx_ids.size()));
        while (!terms.empty() && out.term_ids.size() > cap) {
            terms.pop_back();
            if (const transcribe_status st = encode_terms(); st != TRANSCRIBE_OK) {
                return st;
            }
        }
        char terms_note[96] = "";
        if (terms.size() < terms_in.size()) {
            std::snprintf(terms_note, sizeof(terms_note), "dropped %zu of %zu vocabulary terms",
                          terms_in.size() - terms.size(), terms_in.size());
        }
        char ctx_note[96] = "";
        if (out.ctx_ids.size() < ctx_in) {
            std::snprintf(ctx_note, sizeof(ctx_note), "kept the last %zu of %zu context tokens", out.ctx_ids.size(),
                          ctx_in);
        }
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: %s%s%s (prompt budget: %d tokens)", family, terms_note,
                (terms_note[0] != '\0' && ctx_note[0] != '\0') ? "; " : "", ctx_note, budget);
    }
    out.n_terms = terms.size();
    return TRANSCRIBE_OK;
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
        // One line per prompt: backslash, newline and tab are escaped.
        std::string line;
        for (char c : out) {
            if (c == '\\') {
                line += "\\\\";
            } else if (c == '\n') {
                line += "\\n";
            } else if (c == '\t') {
                line += "\\t";
            } else {
                line += c;
            }
        }
        if (std::FILE * f = std::fopen(path, "ab")) {
            std::fprintf(f, "%s\t%zu\t", family, ids.size());
            std::fwrite(line.data(), 1, line.size(), f);
            std::fputc('\n', f);
            std::fclose(f);
        }
        return;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "%s prompt (%zu tokens): %s", family, ids.size(), out.c_str());
}

}  // namespace transcribe::prompting
