#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER dense_trace_provider

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "tensorflow/core/user_ops/dense-format/dense_trace.h"

#if !defined(_DENSE_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _DENSE_TP_H

#include <lttng/tracepoint.h>

// All of your tracepoint definitions must go before the endif!

TRACEPOINT_EVENT(
                 dense_trace_provider,
                 simple_counter,
                 TP_ARGS(
                         int, int_value
                         ),
                 TP_FIELDS(
                           ctf_integer(int, int_field, int_value)
                           )
)


#endif

#include <lttng/tracepoint-event.h>
