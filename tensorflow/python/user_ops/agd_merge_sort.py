from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as itt

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *

from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes
from tensorflow.python.ops import io_ops, variables, string_ops
from tensorflow.python import user_ops, training as train

import os
import tensorflow as tf

"""
User ops specifically pertaining to agd merge sort

Contains convenience methods for creating common patterns of the user_ops
for agd_merge_sort operation.

You may connect these yourself based on tf.user_ops (the parent module)
"""

def name_generator(base_name, separator="-"):
    """
    Given a basename, defines an op that will generate intermediate unique names
    based on the base_name parameter.

    The suffix will be separated with `separator`, and will start counting from 0.
    """
    start_var = variables.Variable(-1)
    incr_var = start_var.assign_add(1)
    var_as_string = string_ops.as_string(incr_var)
    if not isinstance(base_name, ops.Tensor):
        base_name = constant_op.constant(str(base_name), dtype=dtypes.string,
                                         shape=tensor_shape.scalar(), name="name_generator_base")
    return string_ops.string_join([base_name, var_as_string], separator=separator, name="name_generator")

def _key_maker(file_keys, intermediate_file_prefix, column_grouping_factor, parallel_batches):
    extra_keys = (column_grouping_factor - (len(file_keys) % column_grouping_factor)) % column_grouping_factor
    print("extra keys: {}".format(extra_keys))
    if extra_keys > 0:
        file_keys = list(itt.chain(file_keys, itt.repeat("", extra_keys)))

    string_producer = train.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    batched_output = train.input.batch_pdq([sp_output], batch_size=column_grouping_factor, num_dq_ops=1)
    intermediate_name = name_generator(base_name=intermediate_file_prefix)

    # TODO parallelism can be specified here
    paired_output = train.input.batch_pdq([batched_output[0], intermediate_name], batch_size=1, num_dq_ops=parallel_batches)
    return paired_output

def _make_read_pipeline(key_batch, local_directory, mmap_pool_handle):
    suffix_sep = tf.constant(".")
    base_suffix = tf.constant("base")
    qual_suffix = tf.constant("qual")
    meta_suffix = tf.constant("metadata")
    result_suffix = tf.constant("results")
    bases = []
    quals = []
    metas = []
    results = []
    for k in key_batch:
        bases.append(string_ops.string_join([k, suffix_sep, base_suffix]))
        quals.append(string_ops.string_join([k, suffix_sep, qual_suffix]))
        metas.append(string_ops.string_join([k, suffix_sep, meta_suffix]))
        results.append(string_ops.string_join([k, suffix_sep, result_suffix]))

    # TODO need a buffer pool
    #FIXME if this doesn't work, then you can chain them in with a list like above
    base_reads, base_names = user_ops.FileMMap(filename=bases[0], pool_handle=mmap_pool_handle)
    qual_reads, qual_names = user_ops.FileMMap(filename=quals[0], pool_handle=mmap_pool_handle)
    meta_reads, meta_names = user_ops.FileMMap(filename=metas[0], pool_handle=mmap_pool_handle)
    result_reads, result_names = user_ops.FileMMap(filename=results[0], pool_handle=mmap_pool_handle)

    for b, q, m, r in zip(base_reads[1:], quals[1:], metas[1:], results[1:]):
        base_reads, base_names = user_ops.StagedFileMap(filename=b, upstream_files=base_reads, upstream_name=base_names, pool_handle=mmap_pool_handle)
        qual_reads, qual_names = user_ops.StagedFileMap(filename=q, upstream_files=qual_reads, upstream_name=qual_names, pool_handle=mmap_pool_handle)
        meta_reads, meta_names = user_ops.StagedFileMap(filename=m, upstream_files=meta_reads, upstream_name=meta_names, pool_handle=mmap_pool_handle)
        result_reads, result_names = user_ops.StagedFileMap(filename=r, upstream_files=result_reads, upstream_names=result_names, pool_handle=mmap_pool_handle)

    # TODO need a way to have the AGDReader op process all of these things

    return base_reads, qual_reads, meta_reads, result_reads

def _make_sorters(batch, buffer_list_pool):
    # FIXME this needs the number of records
    for b, q, m, r, num_records, im_name in batch:
        yield user_ops.AGDSort(buffer_list_pool=buffer_list_pool,
                               result_handles=r, base_handels=b,
                               qualities_handles=q, metadata_handles=m,
                               num_records=num_records, name="local_read_agd_sort"), im_name

def _make_agd_batch(ready_batch, buffer_pool):
    for b, q, m, r, inter_name in ready_batch:
        base_reads, base_num_records, base_first_ordinals = user_ops.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=b,
                                                                               name="base_reader")
        qual_reads, qual_num_records, qual_first_ordinals = user_ops.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=q,
                                                                               name="qual_reader")
        meta_reads, meta_num_records, meta_first_ordinals = user_ops.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=m,
                                                                               name="meta_reader")
        result_reads, result_num_records, result_first_ordinals = user_ops.AGDReader(verify=False,
                                                                                     pool_handle=buffer_pool,
                                                                                     file_handle=r,
                                                                                     name="result_reader")
        # TODO we should have some sort of verification on ordinals here!
        yield base_reads, qual_reads, meta_reads, result_reads, base_num_records, inter_name

def _make_writers(results_batch, output_directory):
    for file_handle, num_records, im_name in results_batch:
        writer = user_ops.AGDWriteColumns(record_id="fixme")

# TODO I'm not sure what to do about the last param
def local_read_pipeline(file_keys, local_directory, intermediate_file_prefix="intermediate_file",
                        column_grouping_factor=5, parallel_batches=1, parallel_sort=1):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_process: the parallelism for processing records (reading, decomp)
    parallel_batches: the number of parallel read pipelines
    parallel_sort: the number of parallel sort operations
    """
    if parallel_batches < 1:
        raise Exception("parallel_batches must be >1. Got {}".format(parallel_batches))
    key_producers = _key_maker(file_keys=file_keys, intermediate_file_prefix=intermediate_file_prefix,
                               parallel_batches=parallel_batches)
    mapped_file_pool = user_ops.MMapPool(bound=False, name="local_read_mmap_pool")
    read_pipelines = [(_make_read_pipeline(key_batch=kp[0], local_directory=local_directory, mmap_pool_handle=mapped_file_pool),
                       kp[1]) for kp in key_producers]

    ready_record_batch = train.batch_join_pdq([tuple(k[0])+(k[1],) for k in read_pipelines], num_dp_ops=parallel_process,
                                              batch_size=1, name="ready_record_queue")

    # now the AGD parallel stage
    bp = user_ops.BufferPool(bound=False, name="local_read_buffer_pool")
    processed_record_batch = _make_agd_batch(ready_batch=ready_record_batch, buffer_pool=bp)

    blp = user_ops.BufferListPool(bound=False, name="local_read_buffer_list_pool")

    sorters = _make_sorters(batch=processed_record_batch, buffer_list_pool=blp)

    batched_results = train.batch_join_pdq([a[0] + (a[1],) for a in sorters], num_dq_ops=1,
                                           batch_size=1, name="sorted_im_files_queue")
