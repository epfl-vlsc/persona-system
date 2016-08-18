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

#define DURATION_CALC(_start) (((clock() - _start) / (float) CLOCKS_PER_SEC) * 1000000.0)
#define DURATION_FIELD(_arg_name) ctf_float(double, duration, DURATION_CALC(_arg_name))
#define POINTER_FIELD(_field_name, _arg_name) ctf_integer(uintptr_t, _field_name, (uintptr_t) _arg_name)

TRACEPOINT_EVENT(
                 bioflow,
                 clocks_per_sec,
                 TP_ARGS(),
                 TP_FIELDS(
                           ctf_integer(uint32_t, clocks_per_sec, CLOCKS_PER_SEC)
                           )
                 )

#define POINTER_TIMESTAMP_ARGS TP_ARGS(void*, pointer)
TRACEPOINT_EVENT_CLASS(
                       bioflow,
                       pointer_timestamp,
                       POINTER_TIMESTAMP_ARGS,
                       TP_FIELDS(
                                 POINTER_FIELD(id, pointer)
                                 ctf_integer(clock_t, timestamp, clock())
                                 )
                       )

#define POINTER_TIMESTAMP_EVENT(_name_)               \
  TRACEPOINT_EVENT_INSTANCE(                          \
                            bioflow,                  \
                            pointer_timestamp,        \
                            _name_,                   \
                            POINTER_TIMESTAMP_ARGS    \
                                                    )

#define TIMESTAMP_START_STOP_INSTANCE(_name_)   \
  POINTER_TIMESTAMP_EVENT(_name_ ## _start)     \
  POINTER_TIMESTAMP_EVENT(_name_ ## _stop)

// time spent in the read ready queue (after the reads, and before AGDReaderOp)
TIMESTAMP_START_STOP_INSTANCE(read_ready_queue)

// time spent in the queue of assembled ReadResource instances
TIMESTAMP_START_STOP_INSTANCE(assembled_ready_queue)

// Time that from when the aligner pushes all the subchunks in the work queue, until all these subchunks' results are ready
TIMESTAMP_START_STOP_INSTANCE(total_align)

// Time that a read spends in the work queue waiting to be processed by an aligner thread
TIMESTAMP_START_STOP_INSTANCE(align_ready_queue)

// Time that a BufferList result spends in the queue downstream of the aligner kernel, and upstream of the aligner
TIMESTAMP_START_STOP_INSTANCE(result_ready_queue)

// Time that a BufferList result spends in the queue downstream of the aligner kernel, and upstream of the aligner
TIMESTAMP_START_STOP_INSTANCE(subchunk_time)

TRACEPOINT_EVENT(
                 bioflow,
                 input_processing,
                 TP_ARGS(
                         clock_t, event_start,
                         const void*, input_ptr,
                         const void*, output_ptr
                         ),
                 TP_FIELDS(
                           DURATION_FIELD(event_start)
                           POINTER_FIELD(input_id, input_ptr)
                           POINTER_FIELD(output_id, output_ptr)
                           )
                 )

TRACEPOINT_EVENT(
                 bioflow,
                 read_resource_assembly_no_meta,
                 TP_ARGS(
                         clock_t, event_start,
                         const void*, base_ptr,
                         const void*, qual_ptr
                         ),
                 TP_FIELDS(
                           DURATION_FIELD(event_start)
                           POINTER_FIELD(base_id, base_ptr)
                           POINTER_FIELD(qual_id, qual_ptr)
                           )
                 )

// This event outputs the duration in microseconds

#define AGD_READ_DURATION_ARGS TP_ARGS(clock_t, event_start, uint64_t, first_ordinal, uint32_t, num_records)
TRACEPOINT_EVENT_CLASS(
                        bioflow,
                        agd_read_duration,
                        AGD_READ_DURATION_ARGS,
                        TP_FIELDS(
                                  DURATION_FIELD(event_start)
                                  ctf_integer(uint64_t, first_ordinal, first_ordinal)
                                  ctf_integer(uint32_t, num_records, num_records)
                                  )
                        )

#define BIOFLOW_AGD_READ_DURATION_INSTANCE(_name_)      \
  TRACEPOINT_EVENT_INSTANCE(                              \
                            bioflow,                      \
                            agd_read_duration,          \
                            _name_,                       \
                            AGD_READ_DURATION_ARGS      \
                                                        )

BIOFLOW_AGD_READ_DURATION_INSTANCE(decompression)
BIOFLOW_AGD_READ_DURATION_INSTANCE(base_conversion)

TRACEPOINT_EVENT(
                 bioflow,
                 read_kernel,
                 TP_ARGS(
                         clock_t, event_start,
                         const char*, filename,
                         size_t, file_size
                         ),
                 TP_FIELDS(
                           ctf_integer(clock_t, event_start, event_start)
                           ctf_integer(size_t, size, file_size)
                           DURATION_FIELD(event_start)
                           ctf_string(filename, filename)
                           )
                 )

TRACEPOINT_EVENT(
                 bioflow,
                 snap_alignments,
                 TP_ARGS(
                         clock_t, alignment_start,
                         uint32_t, num_alignments,
                         const void*, result_buf_ptr
                         ),
                 TP_FIELDS(
                           DURATION_FIELD(alignment_start)
                           POINTER_FIELD(result_buf_id, result_buf_ptr)
                           ctf_integer(uint32_t, num_alignments, num_alignments)
                           )
                 )

TRACEPOINT_EVENT(
                 bioflow,
                 snap_align_kernel,
                 TP_ARGS(
                         clock_t, kernel_start,
                         const void*, read_resource_ptr
                         ),
                 TP_FIELDS(
                           DURATION_FIELD(kernel_start)
                           POINTER_FIELD(read_resource_id, read_resource_ptr)
                           )
                 )

TRACEPOINT_EVENT(
                 bioflow,
                 write_duration,
                 TP_ARGS(
                         const char*, filename,
                         uint32_t, num_records
                         ),
                 TP_FIELDS(
                           ctf_integer(clock_t, event_stop, clock())
                           ctf_string(filename, filename)
                           ctf_integer(uint32_t, num_records, num_records)
                           )
                 )

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
