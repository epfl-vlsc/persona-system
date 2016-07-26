#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER bioflow

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "tensorflow/core/user_ops/lttng/tracepoints.h"

#if !defined(_BIOFLOW_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _BIOFLOW_TP_H

#include <lttng/tracepoint.h>
#include <cstdint>
#include <ctime>

// All of your tracepoint definitions must go before the endif!

#define DURATION_CALC(_dur) ((_dur * 1000000) / CLOCKS_PER_SEC)

// This event outputs the duration in microseconds

#define DURATION_ARGS TP_ARGS(clock_t, event_duration)
TRACEPOINT_EVENT_CLASS(
                       bioflow,
                       duration,
                       DURATION_ARGS,
                       TP_FIELDS(
                                 ctf_integer(uint32_t, duration, DURATION_CALC(event_duration))
                                 )
)

#define BIOFLOW_DURATION_INSTANCE(_name_)             \
  TRACEPOINT_EVENT_INSTANCE(                          \
                            bioflow,                  \
                            duration,                 \
                            _name_,                   \
                            DURATION_ARGS             \
                                                    )

#define DENSE_READ_DURATION_ARGS TP_ARGS(clock_t, event_duration, uint64_t, first_ordinal, uint32_t, num_records)
TRACEPOINT_EVENT_CLASS(
                        bioflow,
                        dense_read_duration,
                        DENSE_READ_DURATION_ARGS,
                        TP_FIELDS(
                                  ctf_integer(uint32_t, duration, DURATION_CALC(event_duration))
                                  ctf_integer(uint64_t, first_ordinal, first_ordinal)
                                  ctf_integer(uint32_t, num_records, num_records)
                                  )
                        )

#define BIOFLOW_DENSE_READ_DURATION_INSTANCE(_name_)      \
  TRACEPOINT_EVENT_INSTANCE(                              \
                            bioflow,                      \
                            dense_read_duration,          \
                            _name_,                       \
                            DENSE_READ_DURATION_ARGS      \
                                                        )

BIOFLOW_DENSE_READ_DURATION_INSTANCE(decompression)
BIOFLOW_DENSE_READ_DURATION_INSTANCE(base_conversion)
BIOFLOW_DURATION_INSTANCE(snap_align_kernel)

TRACEPOINT_EVENT(
                 bioflow,
                 file_mmap,
                 TP_ARGS(
                         clock_t, event_duration,
                         const char*, filename
                         ),
                 TP_FIELDS(
                           ctf_integer(uint32_t, duration, DURATION_CALC(event_duration))
                           ctf_string(filename, filename)
                           )
                 )

#endif

#include <lttng/tracepoint-event.h>
