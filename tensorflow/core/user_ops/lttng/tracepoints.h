#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER bioflow_provider

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "tensorflow/core/user_ops/lttng/tracepoints.h"

#if !defined(_BIOFLOW_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _BIOFLOW_TP_H

#include <lttng/tracepoint.h>
#include <cstdint>

// All of your tracepoint definitions must go before the endif!

#define DURATION_ARGS TP_ARGS(uint32_t, event_duration)

TRACEPOINT_EVENT_CLASS(
                       bioflow_provider,
                       duration,
                       DURATION_ARGS,
                       TP_FIELDS(
                                 ctf_integer(uint32_t, duration, event_duration)
                                 )
)

#define BIOFLOW_DURATION_INSTANCE(_name_)         \
  TRACEPOINT_EVENT_INSTANCE(                      \
                            bioflow_provider,     \
                            duration,             \
                            _name_,               \
                            DURATION_ARGS         \
                                                )

BIOFLOW_DURATION_INSTANCE(decompression)
BIOFLOW_DURATION_INSTANCE(file_mmap)

#endif

#include <lttng/tracepoint-event.h>
