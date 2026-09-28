# Prompting

Generic prompting fields on `transcribe_run_params` (CLI flag in parentheses).
The full contract is the comment on those fields in `include/transcribe.h`;
probe `transcribe_model_supports()` for the matching feature bit.

| Field | Feature bit | Effect |
|---|---|---|
| `vocabulary` (`--vocabulary`, `--vocabulary-file`) | `VOCABULARY` | Custom terms, priority order, formatted per family. |
| `prompt` (`--prompt`) | `CONTEXT_PROMPT` | Context text in the model's conditioning slot. |
| `task = INSTRUCT` + `prompt` (`--task instruct`) | `INSTRUCT` | `prompt` replaces the transcription request; output is free text. |
| `prefix` (`--prefix`) | `TRANSCRIPT_PREFIX` | Transcript text the model continues from. Unsupported is an error. |

| Family | Vocabulary | Context prompt | Instruct | Prefix |
|---|---|---|---|---|
| Whisper | yes | yes | | yes |
| Qwen3-ASR | yes | yes | | |
| Voxtral (2507) | | | yes | |
| Granite 4.0-1b / 4.1-2b | yes | | | |
| Granite 4.1-2b-plus | yes | | | yes |
| Canary 180m-flash / 1b-flash / 1b-v2 | | | | yes |
| Fun-ASR-Nano | yes | | | |
| MOSS-Transcribe-Diarize | yes | | | |

Per-model formats and restrictions are in each model doc's **Prompting:**
note under [`models/`](models/).
