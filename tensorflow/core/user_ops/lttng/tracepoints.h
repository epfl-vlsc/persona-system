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

#define DURATION_ARGS TP_ARGS(clock_t, event_duration)

// This event outputs the duration in microseconds
TRACEPOINT_EVENT_CLASS(
                       bioflow,
                       duration,
                       DURATION_ARGS,
                       TP_FIELDS(
                                 ctf_integer(uint32_t, duration, (event_duration * 1000000) / CLOCKS_PER_SEC)
                                 )
)

#define BIOFLOW_DURATION_INSTANCE(_name_)           \
  TRACEPOINT_EVENT_INSTANCE(                        \
                            bioflow,                \
                            duration,               \
                            _name_,                 \
                            DURATION_ARGS           \
                                                  )

BIOFLOW_DURATION_INSTANCE(decompression)
BIOFLOW_DURATION_INSTANCE(file_mmap)
BIOFLOW_DURATION_INSTANCE(base_conversion)
BIOFLOW_DURATION_INSTANCE(snap_align_kernel)

#endif

#include <lttng/tracepoint-event.h>
