# transcribe-cli routes a DIARIZE model to the diarize path and prints speaker
# segments for the 2-speaker oracle mix. Gated on TRANSCRIBE_SORTFORMER_GGUF (reported as skipped).

if("$ENV{TRANSCRIBE_SORTFORMER_GGUF}" STREQUAL "")
    message("SKIP: TRANSCRIBE_SORTFORMER_GGUF not set")
    return()
endif()

execute_process(
    COMMAND "${CLI}" -q --backend cpu -m "$ENV{TRANSCRIBE_SORTFORMER_GGUF}" "${WAV}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "transcribe-cli exited ${rc}\n${out}\n${err}")
endif()
if(NOT out MATCHES "speaker segments: [1-9].*S2")
    message(FATAL_ERROR "unexpected output:\n${out}")
endif()
