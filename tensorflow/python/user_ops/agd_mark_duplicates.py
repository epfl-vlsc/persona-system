from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *

from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops
from tensorflow.python import user_ops, training as train
from tensorflow.python.training import queue_runner
from tensorflow.python.user_ops import user_ops as uop

import os
import tensorflow as tf


"""
User ops specifically pertaining to agd mark duplicates

Contains convenience methods for creating common patterns for 
marking duplicates.

You may connect these yourself based on tf.user_ops (the parent module)
"""

def _key_maker(file_keys, parallel_read):
    num_file_keys = len(file_keys)

    string_producer = train.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    keys = train.input.batch_pdq([sp_output], batch_size=1, num_dq_ops=parallel_read, name="key_queue")

    return keys

def _make_read_pipeline(keys, local_directory, mmap_pool_handle, bp_handle):

    suffix_sep = tf.constant(".")
    result_suffix = tf.constant("results")
    to_enqueue = []

    for key in keys:

        results_name = string_ops.string_join(key, suffix_sep, result_suffix)

        results_raw, names = uop.FileMMap(filename=results_name, handle=mmap_pool_handle, local_prefix=local_directory, synchronous=True, name="results_mmap")

        results_r = results_raw.expand_dims(0)
        results, num_results, first_ordinal = uop.AGDReader(file_handle=results_r, pool_handle=bp_handle, name="results_reader")

        print("results: {}".format(results))
        results_ready = tf.unpack(results, name="result_unpack")
        print("results: {}".format(results_ready))
        print("numresults: {}".format(num_results))
        print("first ord: {}".format(first_ordinal))
        to_enqueue.append([results_ready[0], num_results, results_name, first_ordinal])

    result_queue = tf.train.batch_join_pdq(to_enqueue, batch_size=1,
                                      enqueue_many=False,
                                      num_dq_ops=1,
                                      name="result_ready_queue")

    return result_queue # [result_handle, num_results, filename, first_ord]


def _make_writers(results_batch, output_dir):
    for column_handle, num_records, name, first_ord in results_batch:
        writer, first_o_passthru = uop.AGDWriteColumns(record_id="fixme",
                                                       record_type=["results"],
                                                       column_handle=column_handle,
                                                       compress=False,
                                                       output_dir=output_dir + "/",
                                                       file_path=name,
                                                       first_ordinal=first_ord,
                                                       num_records=num_records,
                                                       name="agd_column_writer")
        yield writer # writes out the file path key (full path)


def agd_mark_duplicates_local(file_keys, local_directory, outdir=None, parallel_read=1, parallel_write=1):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_process: the parallelism for processing records (reading, decomp)
    parallel_read: the number of parallel read pipelines
    parallel_sort: the number of parallel sort operations
    """
    if parallel_read < 1:
        raise Exception("parallel_read must be >1. Got {}".format(parallel_read))

    key_producers = _key_maker(file_keys=file_keys, parallel_read=parallel_read)
    mapped_file_pool = uop.MMapPool(bound=False, name="local_read_mmap_pool")
    bp = uop.BufferPool(bound=False, name="local_read_buffer_pool")
   
    results_pipe = _make_read_pipeline(key_producers, local_directory, mmap_file_pool, bp) # [result_handle, num_results, filename, first_ord]
    
    result_handle = results_pipe[0]
    num_results = results_pipe[1]
    filename = results_pipe[2]
    first_ord = results_pipe[3]

    result_out = uop.AGDMarkDuplicates(results_handle=result_handle, num_records=num_results, 
            buffer_list_pool=bp, name="markdupsop")

    result_to_write = tf.train.batch_pdq([result_out, num_results, filename, first_ord],
                                        batch_size=1, num_dq_ops=parallel_write, name="to_write_queue")


    written = _make_writers(results_batch=result_to_write, output_dir=outdir)

    recs = [rec for rec in written]
    all_written_keys = train.input.batch_pdq(recs, num_dq_ops=1,
                                        batch_size=1, name="written_key_out")

    print("all written keys: {}".format(all_written_keys))
    return all_written_keys
  
  
