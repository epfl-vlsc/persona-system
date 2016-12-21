from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as itt

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

def _key_maker(file_keys, intermediate_file_prefix, column_grouping_factor, parallel_read):
    extra_keys = (column_grouping_factor - (len(file_keys) % column_grouping_factor)) % column_grouping_factor
    print("extra keys: {}".format(extra_keys))
    if extra_keys > 0:
        file_keys = list(itt.chain(file_keys, itt.repeat("", extra_keys)))

    string_producer = train.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    batched_output = train.input.batch_pdq([sp_output], batch_size=column_grouping_factor, num_dq_ops=1)
    if column_grouping_factor == 1:
        batched_output = [tf.pack((bo,)) for bo in batched_output]
    intermediate_name = name_generator(base_name=intermediate_file_prefix)

    # TODO parallelism can be specified here
    paired_output = train.input.batch_pdq([batched_output[0], intermediate_name], batch_size=1, num_dq_ops=parallel_read)
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
    split_batch = array_ops.unpack(key_batch)
    for k in split_batch:
        bases.append(string_ops.string_join([k, suffix_sep, base_suffix]))
        quals.append(string_ops.string_join([k, suffix_sep, qual_suffix]))
        metas.append(string_ops.string_join([k, suffix_sep, meta_suffix]))
        results.append(string_ops.string_join([k, suffix_sep, result_suffix]))

    # TODO if this doesn't work, then you can chain them in with a list like above
    base_reads, base_names = uop.FileMMap(filename=bases[0], handle=mmap_pool_handle, local_prefix=local_directory, name="base_mmap")
    qual_reads, qual_names = uop.FileMMap(filename=quals[0], handle=mmap_pool_handle, local_prefix=local_directory, name="qual_mmap")
    meta_reads, meta_names = uop.FileMMap(filename=metas[0], handle=mmap_pool_handle, local_prefix=local_directory, name="meta_mmap")
    result_reads, result_names = uop.FileMMap(filename=results[0], handle=mmap_pool_handle, local_prefix=local_directory, name="result_mmap")

    for b, q, m, r in zip(bases[1:], quals[1:], metas[1:], results[1:]):
        base_reads, base_names = uop.StagedFileMap(filename=b, upstream_files=base_reads, upstream_names=base_names, handle=mmap_pool_handle,
                                                        local_prefix=local_directory, name="base_staged_mmap")
        qual_reads, qual_names = uop.StagedFileMap(filename=q, upstream_files=qual_reads, upstream_names=qual_names, handle=mmap_pool_handle,
                                                        local_prefix=local_directory, name="qual_staged_mmap")
        meta_reads, meta_names = uop.StagedFileMap(filename=m, upstream_files=meta_reads, upstream_names=meta_names, handle=mmap_pool_handle,
                                                        local_prefix=local_directory, name="meta_staged_mmap")
        result_reads, result_names = uop.StagedFileMap(filename=r, upstream_files=result_reads, upstream_names=result_names, handle=mmap_pool_handle,
                                                            local_prefix=local_directory, name="result_staged_mmap")

    return base_reads, qual_reads, meta_reads, result_reads

def _make_sorters(batch, buffer_list_pool):
    # FIXME this needs the number of records
    for b, q, m, r, num_records, im_name in batch:
        yield uop.AGDSort(buffer_list_pool=buffer_list_pool,
                          results_handles=r, bases_handles=b,
                          qualities_handles=q, metadata_handles=m,
                          num_records=num_records, name="local_read_agd_sort"), im_name

def _make_agd_batch(ready_batch, buffer_pool):
    for b, q, m, r, inter_name in ready_batch:
        base_reads, base_num_records, base_first_ordinals = uop.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=b,
                                                                               name="base_reader")
        qual_reads, qual_num_records, qual_first_ordinals = uop.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=q,
                                                                               name="qual_reader")
        meta_reads, meta_num_records, meta_first_ordinals = uop.AGDReader(verify=False,
                                                                               pool_handle=buffer_pool,
                                                                               file_handle=m,
                                                                               name="meta_reader")
        result_reads, result_num_records, result_first_ordinals = uop.AGDReader(verify=False,
                                                                                     pool_handle=buffer_pool,
                                                                                     file_handle=r,
                                                                                     name="result_reader")
        # TODO we should have some sort of verification on ordinals here!
        yield base_reads, qual_reads, meta_reads, result_reads, base_num_records, inter_name

def _make_writers(results_batch, output_dir):
    first_ordinal = constant_op.constant(0, dtype=dtypes.int64) # first ordinal doesn't matter for the sort phase
    for column_handle, num_records, im_name in results_batch:
        writer, first_o_passthru = uop.AGDWriteColumns(record_id="fixme",
                                                       record_type=["base", "qual", "metadata", "results"],
                                                       column_handle=column_handle,
                                                       compress=False,
                                                       output_dir=output_dir + "/",
                                                       file_path=im_name,
                                                       first_ordinal=first_ordinal,
                                                       num_records=num_records,
                                                       name="agd_column_writer")
        yield writer # writes out the file path key (full path)

# TODO I'm not sure what to do about the last param
def local_sort_pipeline(file_keys, local_directory, outdir=None, intermediate_file_prefix="intermediate_file",
                        column_grouping_factor=5, parallel_read=1, parallel_process=1, parallel_sort=1):
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
    key_producers = _key_maker(file_keys=file_keys, intermediate_file_prefix=intermediate_file_prefix,
                               parallel_read=parallel_read, column_grouping_factor=column_grouping_factor)
    mapped_file_pool = uop.MMapPool(bound=False, name="local_read_mmap_pool")
    read_pipelines = [(_make_read_pipeline(key_batch=kp[0], local_directory=local_directory, mmap_pool_handle=mapped_file_pool),
                       kp[1]) for kp in key_producers]

    ready_record_batch = train.input.batch_join_pdq([tuple(k[0])+(k[1],) for k in read_pipelines], num_dq_ops=parallel_process,
                                                    batch_size=1, name="ready_record_queue")

    # now the AGD parallel stage
    bp = uop.BufferPool(bound=False, name="local_read_buffer_pool")
    processed_record_batch = _make_agd_batch(ready_batch=ready_record_batch, buffer_pool=bp)

    blp = uop.BufferListPool(bound=False, name="local_read_buffer_list_pool")

    sorters = _make_sorters(batch=processed_record_batch, buffer_list_pool=blp)

    batched_results = train.input.batch_join_pdq([a[0] + (a[1],) for a in sorters], num_dq_ops=1,
                                                 batch_size=1, name="sorted_im_files_queue")

    if outdir is None:
        outdir = local_directory
    intermediate_keys = _make_writers(results_batch=batched_results, output_dir=outdir)

    all_im_keys = train.input.batch_join_pdq([(im_key,) for im_key in intermediate_keys], num_dq_ops=1,
                                             batch_size=1, name="intermediate_key_queue")

    return all_im_keys

### All the methods for creating the local merge pipeline

def _make_merge_read_records(key_outs, in_dir, mmap_pool_handle):
    suffix_sep = tf.constant(".")
    base_suffix = tf.constant("base")
    qual_suffix = tf.constant("qual")
    meta_suffix = tf.constant("metadata")
    result_suffix = tf.constant("results")
    # dictated by the merge op
    suffix_order = [result_suffix, base_suffix, qual_suffix, meta_suffix]

    def make_single_chunk_read(im_name):
        appended_names = [string_ops.string_join([im_name, suffix_sep, a]) for a in suffix_order]
        reads, names = uop.FileMMap(filename=appended_names[0], local_prefix=in_dir,
                                    handle=mmap_pool_handle, name="result_mmap")
        accum = [(reads, names)]
        for column in appended_names[1:]:
            prior_reads, prior_names = accum[-1]
            #import ipdb; ipdb.set_trace()
            reads, names = uop.StagedFileMap(filename=column,
                                             upstream_files=prior_reads,
                                             upstream_names=prior_names,
                                             handle=mmap_pool_handle,
                                             local_prefix=in_dir,
                                             name="merge_column_mmap")
            accum.append((reads, names))
        return accum[-1][0]

    for key_out in key_outs:
        split_records = array_ops.unpack(key_out)
        yield [make_single_chunk_read(im_name=im_name) for im_name in split_records]

def _make_processed_records(ready_read_records, buffer_pool):
    def process_ready_row(interm_columns):
        columns_split = tf.unpack(interm_columns)
        return zip(*(uop.AGDReader(verify=False,
                                   pool_handle=buffer_pool,
                                   file_handle=column,
                                   name="column_agd_reader") for column in columns_split))


    for interm_columns in ready_read_records:
        readss, num_recordss, first_ordinalss = process_ready_row(interm_columns=interm_columns)
        yield [nr[0] for nr in num_recordss], tf.pack(readss)

def local_merge_pipeline(intermediate_keys, in_dir, record_name, outdir=None, chunk_size=100000):
    if chunk_size < 1:
        raise Exception("Need strictly non-negative chunk size. Got {}".format(chunk_size))
    key_producer = train.input.input_producer([intermediate_keys],
                                              # this element_shape specification isn't necessary, but it's a good double-check
                                              element_shape=tensor_shape.vector(len(intermediate_keys)),
                                              capacity=1,
                                              shuffle=False,
                                              num_epochs=1,
                                              name="merge_key_producer")
    key_output = key_producer.dequeue()
    key_outs = train.input.batch_pdq([key_output], batch_size=1, num_dq_ops=1)
    mapped_file_pool = uop.MMapPool(bound=False, name="local_read_mmap_pool")
    ready_read_records = _make_merge_read_records(key_outs=key_outs, in_dir=in_dir,
                                                  mmap_pool_handle=mapped_file_pool)

    bp = uop.BufferPool(bound=False, name="local_read_merge_buffer_pool")
    processed_records = _make_processed_records(ready_read_records=ready_read_records, buffer_pool=bp)

    merge_ready_queue = train.input.batch_join_pdq([p for p in processed_records],
                                                   num_dq_ops=1, batch_size=1, name="merge_ready_queue")
    q = data_flow_ops.FIFOQueue(capacity=100000, # big because who cares
                                dtypes=[dtypes.int32, dtypes.string],
                                shapes=[tensor_shape.scalar(), tensor_shape.vector(2)],
                                name="merge_output_queue")

    blp = uop.BufferListPool(bound=False, name="local_read_merge_buffer_list_pool")

    num_records, chunk_group_handles = merge_ready_queue[0]
    merge_op = uop.AGDMerge(chunk_size=chunk_size,
                            buffer_list_pool=blp,
                            num_records=num_records,
                            chunk_group_handles=chunk_group_handles,
                            output_buffer_queue_handle=q.queue_ref)
    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [merge_op]))

    if outdir is None:
        outdir = in_dir

    # FIXME if you don't have at least chunk_size records in your dataset, this will cause underflow
    # this is a hack!
    first_ordinal = tf.Variable(-1 * chunk_size, dtype=dtypes.int64, name="first_ordinal")

    record_name_constant = constant_op.constant(record_name+"-")
    num_recs, buffer_list_handle = q.dequeue()
    first_ord = first_ordinal.assign_add(math_ops.to_int64(num_recs, name="first_ord_cast_to_64"), use_locking=True)
    first_ord_str = string_ops.as_string(first_ord, name="first_ord_string")
    file_name = string_ops.string_join([record_name_constant, first_ord_str], name="file_name_string_joiner")
    write_join_tensor = control_flow_ops.tuple(tensors=[buffer_list_handle, num_recs, first_ord, file_name], name="write_join_tensor")
    write_join_queue = train.input.batch_pdq(write_join_tensor, num_dq_ops=1, batch_size=1, name="write_join_queue", capacity=1)

    final_write_out = []
    for buff_list, n_recs, first_o, file_key in write_join_queue:
        file_key_passthru, first_o_passthru = uop.AGDWriteColumns(record_id=record_name,
                                                                  record_type=["results", "base", "qual", "metadata"],
                                                                  column_handle=buff_list,
                                                                  compress=False,
                                                                  output_dir=outdir+"/",
                                                                  file_path=file_key,
                                                                  first_ordinal=first_o,
                                                                  num_records=n_recs,
                                                                  name="agd_column_writer_merge")
        final_write_out.append([file_key_passthru, first_o_passthru, n_recs])
    #return final_write_out[0]

    sink_queue = train.input.batch_join_pdq(final_write_out, capacity=1, num_dq_ops=1, batch_size=1, name="final_sink_queue")
    return sink_queue[0]
