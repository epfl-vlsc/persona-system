#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER bioflow

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "tensorflow/core/user_ops/lttng/tracepoints.h"

#if !defined(_BIOFLOW_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _BIOFLOW_TP_H

#include <lttng/tracepoint.h>
#include <cstdint>
#include <chrono>

// All of your tracepoint definitions must go before the endif!

#define DURATION_FIELD(_arg_name) ctf_integer(uint64_t, duration, _arg_name.count())
#define POINTER_FIELD(_field_name, _arg_name) ctf_integer(uintptr_t, _field_name, (uintptr_t) _arg_name)

#define STRING_DURATION_ARGS TP_ARGS(const char*, s, \
                                     std::chrono::microseconds, duration)

TRACEPOINT_EVENT_CLASS(
                       bioflow,
                       string_duration,
                       STRING_DURATION_ARGS,
                       TP_FIELDS(
                                 ctf_string(string_val, s)
                                 DURATION_FIELD(duration)
                                 )
                       )

#define STRING_DURATION_INSTANCE(_name)             \
  TRACEPOINT_EVENT_INSTANCE(                        \
                            bioflow,                \
                            string_duration,        \
                            _name,                  \
                            STRING_DURATION_ARGS )

STRING_DURATION_INSTANCE(chunk_read)
STRING_DURATION_INSTANCE(chunk_write)

// This event outputs the duration in microseconds

TRACEPOINT_EVENT(
                 bioflow,
                 reads_aligned,
                 TP_ARGS(
                         uint32_t, num_reads,
                         int, thread_id,
                         const void*, aligner_ptr
                         ),
                 TP_FIELDS(
                           ctf_integer(uint32_t, num_reads, num_reads)
                           ctf_integer(int, thread_id, thread_id)
                           POINTER_FIELD(aligner_id, aligner_ptr)
                           )
                 )

TRACEPOINT_EVENT(
                 bioflow,
                 chunk_aligned,
                 TP_ARGS(std::chrono::microseconds, duration),
                 TP_FIELDS(
                           DURATION_FIELD(duration)
                           )
                 )

// Used to check when the first key starts (to get accurate start info, to subtract from Tensorflow startup stuff)
TRACEPOINT_EVENT(
                 bioflow,
                 process_key,
                 TP_ARGS(
                         const char*, filename
                         ),
                 TP_FIELDS(
                           ctf_string(filename, filename)
                           )
                 )

#endif

#include <lttng/tracepoint-event.h>
