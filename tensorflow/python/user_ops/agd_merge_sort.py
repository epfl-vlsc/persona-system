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

def _key_maker(file_keys, intermediate_file_prefix, parallel_merge_sort, parallel_batches):
    extra_keys = (parallel_merge_sort - (len(file_keys) % parallel_merge_sort)) % parallel_merge_sort
    print("extra keys: {}".format(extra_keys))
    if extra_keys > 0:
        file_keys = list(itt.chain(file_keys, itt.repeat("", extra_keys)))

    string_producer = train.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    batched_output = train.input.batch_pdq([sp_output], batch_size=parallel_merge_sort, num_dq_ops=1)
    intermediate_name = name_generator(base_name=intermediate_file_prefix)

    # TODO parallelism can be specified here
    paired_output = train.input.batch_pdq([batched_output[0], intermediate_name], batch_size=1, num_dq_ops=1)
    return paired_output

def _make_read_pipeline(key_batch, local_directory):
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
    base_reads, base_names = user_ops.FileMMap(filename=bases[0])
    qual_reads, qual_names = user_ops.FileMMap(filename=quals[0])
    meta_reads, meta_names = user_ops.FileMMap(filename=metas[0])
    result_reads, result_names = user_ops.FileMMap(filename=results[0])

    for b, q, m, r in zip(base_reads[1:], quals[1:], metas[1:], results[1:]):
        base_reads, base_names = user_ops.StagedFileMap(filename=b, upstream_files=base_reads, upstream_name=base_names)
        qual_reads, qual_names = user_ops.StagedFileMap(filename=q, upstream_files=qual_reads, upstream_name=qual_names)
        meta_reads, meta_names = user_ops.StagedFileMap(filename=m, upstream_files=meta_reads, upstream_name=meta_names)
        result_reads, result_names = user_ops.StagedFileMap(filename=r, upstream_files=result_reads, upstream_names=result_names)

    return base_reads, qual_reads, meta_reads, result_reads

# TODO I'm not sure what to do about the last param
def local_read_pipeline(file_keys, local_directory, intermediate_file_prefix="intermediate_file",
                        parallel_merge_sort=5, parallel_batches=1):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    local_directory: the "base path" from which these should be read
    parallel_merge_sort: the number of keys to put together
    """
    if parallel_batches < 1:
        raise Exception("parallel_batches must be >1. Got {}".format(parallel_batches))
    key_producers = _key_maker(file_keys=file_keys, intermediate_file_prefix=intermediate_file_prefix,
                               parallel_batches=parallel_batches)
    read_pipelines = [(_make_read_pipeline(key_batch=kp[0]), kp[1]) for kp in key_producers]

