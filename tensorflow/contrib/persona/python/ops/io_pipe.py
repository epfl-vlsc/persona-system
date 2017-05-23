
# build input pipelines for AGD

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.persona.python.ops.persona_ops import persona_ops as persona_ops_proxy
from tensorflow.contrib.persona.python.ops.queues import batch_pdq, batch_join_pdq
from tensorflow.python.framework import ops, dtypes, tensor_shape, constant_op
from tensorflow.python.ops import string_ops, array_ops
from tensorflow.python import training
from tensorflow.python.training.input import batch

import json
import os
import functools

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


persona_ops = persona_ops_proxy()
scalar_shape = tensor_shape.scalar()
resource_shape = tensor_shape.vector(2)
pool_default_args = {'size':10, "bound": False}
suffix_separator = constant_op.constant(".")
default_records_type = ({"type": "structured", "extension": "results"},)
def check_valid_record_type(record_type):
    if not ("type" in record_type and "extension" in record_type and isinstance(record_type, dict)):
        raise Exception("Invalid record type: {}".format(record_type))

def validate_shape_and_dtype(tensor, expected_shape, expected_dtype):
    tensor_shape = tensor.get_shape()
    if tensor_shape != expected_shape:
        raise Exception("Tensor {t} doesn't meet expected shape {s}. Has {a}".format(t=tensor, s=expected_shape, a=tensor_shape))
    tensor_dtype = tensor.dtype
    if tensor_dtype != expected_dtype:
        raise Exception("Tensor {t} doesn't have expected dtype {d}. Has {a}".format(t=tensor, d=expected_dtype, a=tensor_dtype))

valid_columns = {"base", "qual", "metadata", "results"} # TODO add for other columns
def validate_columns(columns):
    """
    Validates the columns based on their validity, returning a set
    :param columns: 
    :return: a set of columns, constructed from the iterable passed in as the param
    """
    if len(columns) == 0:
        raise Exception("Ceph Read Pipeline must read >0 columns")
    new_columns = set(columns)

    invalid_columns = new_columns - valid_columns
    if len(invalid_columns) != 0:
        raise Exception("Can't instantiate Ceph Read Pipeline with invalid columns: {}".format(invalid_columns))

    return columns

def expand_column_extensions(key, columns):
    """
    Expands a given AGD key into the full extensions, based on the columns
    :param keys: an iterator of scalar strings, representing the keys for a given parallelism level
    :param columns: assumed to have been validated previously be the caller
    :yield: a generator for keys
    """
    for c in columns:
        yield string_ops.string_join(inputs=[key, c], separator=".", name="AGD_column_expansion")

def ceph_read_pipeline(upstream_tensors, user_name, cluster_name, ceph_conf_path, columns, pool_name,
                       ceph_read_size=2**26, buffer_pool=None, buffer_pool_args=pool_default_args, name="ceph_read_pipeline"):
    """
    Create a ceph input pipeline.
    
    FIXME doesn't return the generic column name that was read. Must be assumed to be in order based on the columns
    :param upstream_tensors: a tuple of tensors (key, pool_name), which are typically found in the metadata file. This controls the parallelism
    :param user_name: 
    :param cluster_name: 
    :param ceph_conf_path: 
    :param columns: 
    :param downstream_parallel: the level of parallelism to create for the downstream nodes
    :param ceph_read_size: 
    :param buffer_pool: 
    :param name: 
    :return: a list of (key, pool_name, tuple(chunk_buffers)) for every tensor in upstream tensors
    """
    def make_ceph_reader(key, namespace):
        return persona_ops.ceph_reader(cluster_name=cluster_name,
                                       user_name=user_name,
                                       namespace=namespace,
                                       pool_name=pool_name,
                                       ceph_conf_path=ceph_conf_path,
                                       read_size=ceph_read_size,
                                       key=key,
                                       buffer_pool=buffer_pool) # buffer_pool is in scope. hooray python!
    columns = validate_columns(columns=columns)

    if buffer_pool is None:
        buffer_pool = persona_ops.buffer_pool(**buffer_pool_args)

    for key, namespace in upstream_tensors:
        validate_shape_and_dtype(tensor=key, expected_shape=scalar_shape, expected_dtype=dtypes.string)
        validate_shape_and_dtype(tensor=namespace, expected_shape=scalar_shape, expected_dtype=dtypes.string)
        chunk_buffers = tuple(make_ceph_reader(key=column_key, namespace=namespace) for column_key in expand_column_extensions(key=key, columns=columns))
        yield key, namespace, chunk_buffers

def aligner_compress_pipeline(upstream_tensors, buffer_pool=None, buffer_pool_args=pool_default_args, name="aligner_compress_pipeline"):
    """
    Compresses a list of upstream tensors of buffer list (via handles) into buffers
    :param upstream_tensors: 
    :param name: 
    :return: 
    """
    def compress_buffer_list(buffer_list):
        return persona_ops.buffer_list_compressor(buffer_list=buffer_list, buffer_pool=buffer_pool)
    if buffer_pool is None:
        buffer_pool = persona_ops.buffer_pool(**buffer_pool_args)
    for buffer_lists in upstream_tensors:
        bls_unstacked = array_ops.unstack(buffer_lists)
        compressed_buffers = tuple(compress_buffer_list(a) for a in bls_unstacked)
        yield array_ops.stack(compressed_buffers)

# note: upstream tensors has the record_id, namespace, key, and chunk_buffers. Mark this in the doc
def ceph_aligner_write_pipeline(upstream_tensors, user_name, cluster_name, pool_name, ceph_conf_path, compressed, record_types=default_records_type, name="ceph_write_pipeline"):
    """
    Create a ceph write pipeline for aligner results (that outputs a BufferList, which we must wait on for completion
    :param upstream_tensors: a list of aligner output tensors of type (key, first ordinal, number of records, pool name, record id, column handle)
    :param user_name: 
    :param cluster_name: 
    :param ceph_conf_path: 
    :param name: 
    :return: yields the output of ceph write columns
    """
    if compressed:
        writer_op = functools.partial(persona_ops.agd_ceph_buffer_writer, compressed=True)
    else:
        writer_op = persona_ops.agd_ceph_buffer_list_writer
    def make_ceph_writer(key, first_ordinal, num_records, column_handle, namespace, record_id):
        column_handles = array_ops.unstack(column_handle)
        if not len(column_handles) == len(record_types):
            raise Exception("number of record types ({r}) must be equal to number of columns ({c})".format(r=len(record_types),
                                                                                                           c=len(column_handles)))
        for handle, record_type in zip(column_handles, record_types):
            check_valid_record_type(record_type=record_type)
            full_key = string_ops.string_join([key, suffix_separator, record_type["extension"]])
            yield writer_op(cluster_name=cluster_name,
                             user_name=user_name,
                             ceph_conf_path=ceph_conf_path,
                             pool_name=pool_name,
                             namespace=namespace,
                             record_id=record_id,
                             record_type=record_type["type"],
                             num_records=num_records,
                             first_ordinal=first_ordinal,
                             path=full_key,
                             resource_handle=handle)

    for key, namespace, num_records, first_ordinal, record_id, column_handle in upstream_tensors:
        yield make_ceph_writer(key=key,
                               first_ordinal=first_ordinal,
                               num_records=num_records,
                               record_id=record_id,
                               namespace=namespace,
                               column_handle=column_handle)

def local_read_pipeline(upstream_tensors, columns, sync=True, mmap_pool=None, mmap_pool_args=pool_default_args, name="local_read_pipeline"):
    """
    Create a read pipeline to read from the filesystem
    :param upstream_tensors: a list of file keys, as extracted from the metadata file
    :param columns: a list of columns to extract. See `valid_columns` for the set of valid columns
    :param sync: whether or not to synchronously map the files
    :param mmap_pool: if not None, provide a persona_ops.file_m_map pool to this method
    :param mmap_pool_args:
    :param name: 
    :return: yield a tuple of '(persona_ops.file_m_map for every column file, a generator)' for every tensor in upstream_tensors
    """
    def make_readers(input_file_basename):
        prev = []
        for full_filename in expand_column_extensions(key=input_file_basename, columns=columns):
            with ops.control_dependencies(prev):
                mmap_op = persona_ops.file_m_map(filename=full_filename, pool_handle=mmap_pool, synchronous=sync)
                yield mmap_op
                prev.append(mmap_op)

    columns = validate_columns(columns=columns)

    if mmap_pool is None:
        mmap_pool = persona_ops.m_map_pool(name=name, **mmap_pool_args)

    assert len(upstream_tensors) > 0
    for file_path in upstream_tensors:
        yield make_readers(input_file_basename=file_path)

def local_write_pipeline(upstream_tensors, compressed, record_types=default_records_type, name="local_write_pipeline"):
    """
    Create a local write pipeline, based on the number of upstream tensors received.
    :param upstream_tensors: a list of tensor tuples of type: buffer_list_handle, record_id, first_ordinal, num_records, file_path
    :param record_type: the type of results to write. See persona_ops.cc for valid types
    :param name: 
    :return: yield a writer for each record to be written in upstream tensors
    """
    if compressed:
        writer_op = functools.partial(persona_ops.agd_file_system_buffer_writer, compressed=True)
    else:
        writer_op = persona_ops.agd_file_system_buffer_list_writer
    def make_writer(record_id, file_path, first_ordinal, num_records, bl_handle):
        bl_handle = array_ops.unstack(bl_handle)
        if not len(bl_handle) == len(record_types):
            raise Exception("number of record types must equal number of buffer list handles")
        for handle, record_type in zip(bl_handle, record_types):
            check_valid_record_type(record_type=record_type)
            full_filepath = string_ops.string_join([file_path, suffix_separator, record_type["extension"]])
            yield writer_op(record_id=record_id,
                            record_type=record_type["type"],
                            resource_handle=handle,
                            first_ordinal=first_ordinal,
                            num_records=num_records,
                            path=full_filepath)

    assert len(upstream_tensors) > 0
    for buffer_list_handle, record_id, first_ordinal, num_records, file_path in upstream_tensors:
        yield make_writer(record_id=record_id,
                          file_path=file_path,
                          num_records=num_records,
                          first_ordinal=first_ordinal,
                          bl_handle=buffer_list_handle)

def agd_reader_pipeline(upstream_tensors, verify=False, buffer_pool=None, buffer_pool_args=pool_default_args, name="agd_reader_pipeline"):
    """
    Yield a pipeline of input buffers processed by AGDReader.
    
    This processes ONLY A SINGLE COLUMN. Use agd_reader_multi_column_pipeline to do multiple columns in parallel.
    
    :param upstream_tensors: a tensor of handles to resources of type Data (in C++ persona code)
    :param verify: if True, enable format verification by AGDReader. Will fail if shape doesn't conform, but causes performance impact
    :param buffer_pool: if not None, use this as the buffer_pool, else create buffer_pool
    :param buffer_pool_default_args: the arguments to make the buffer_pool, if it is None
    :param name: 
    :return: yields a tuple of output_buffer, num_records, first_ordinal, record_id
    """
    if buffer_pool is None:
        buffer_pool = persona_ops.buffer_pool(**buffer_pool_args, name="agd_reader_buffer_pool")
    if isinstance(upstream_tensors, ops.Tensor):
        upstream_tensors = array_ops.unstack(upstream_tensors)
    assert len(upstream_tensors) > 0
    for upstream_tensor in upstream_tensors:
        ut_shape = upstream_tensor.get_shape()
        if ut_shape != resource_shape:
            raise Exception("AGD_Reader pipeline encounter Tensor with shape {actual}, but expected {expected}".format(
                actual=ut_shape, expected=resource_shape
            ))
        output_buffer, num_records, first_ordinal, record_id = persona_ops.agd_reader(buffer_pool=buffer_pool, file_handle=upstream_tensor,
                                                                                      verify=verify, unpack=True, name="agd_reader")
        yield output_buffer, num_records, first_ordinal, record_id

def agd_reader_multi_column_pipeline(upstream_tensorz, verify=False, buffer_pool=None, share_buffer_pool=True, buffer_pool_args=pool_default_args, name="agd_reader_multi_column_pipeline"):
    """
    Create an AGDReader pipeline for an iterable of columns. Each column group is assumed to have the same first ordinal, number of records, and record id.
    :param upstream_tensorz: a list of list of tensors, each item being a column group
    :param verify: whether or not to invoke the verification for AGD columns
    :param buffer_pool: pass in a buffer_pool to reuse
    :param share_buffer_pool: if buffer_pool is not passed in, create one to share among all the AGDReader instances
    :param buffer_pool_args: special buffer pool args, if it's created
    :param name: 
    :return: yield [output_buffer_handles], num_records, first_ordinal, record_id; in order, for each column group in upstream_tensorz
    """
    if buffer_pool is None and share_buffer_pool:
        buffer_pool = persona_ops.buffer_pool(**buffer_pool_args, name="agd_reader_buffer_pool")
    assert len(upstream_tensorz) > 0
    process_tensorz = (agd_reader_pipeline(upstream_tensors=upstream_tensors, verify=verify, buffer_pool_args=buffer_pool_args, buffer_pool=buffer_pool)
                       for upstream_tensors in upstream_tensorz)
    for processed_tensors in process_tensorz:
        output_buffers, num_recordss, first_ordinalss, record_ids = zip(*processed_tensors)
        yield output_buffers, num_recordss[0], first_ordinalss[0], record_ids[0]

def agd_bwa_read_assembler(upstream_tensors, agd_read_pool=None, agd_read_pool_args=pool_default_args, include_meta=False, name="agd_read_assembler"):
    """
    Generate agd_bwa_read datatypes from the upstream tensors. BWA paired aligner requires specific data structures
    :param upstream_tensors: a list of tuples of tensors with type: (column_buffers, num_reads)
    :param agd_read_pool: if not None, pass in an instance of persona_ops.agd_read_pool to share
    :param agd_read_pool_args: args for deafult construction of agd_read_pool if it's None
    :param include_meta: create a meta read assembler if passed. The shape of upstream_tensors must be compatible
    :param name: 
    :return: yield instances of a tensor with AGDRead instance as the result
    """
    def make_agd_read(column_buffers, num_reads):
        # order on column_buffers: bases, quals, metadata (if exists)
        if isinstance(column_buffers, ops.Tensor):
            column_buffers = array_ops.unstack(column_buffers)
        if include_meta:
            assert len(column_buffers) == 3
            return persona_ops.bwa_assembler(bwa_read_pool=agd_read_pool,
                                             base_handle=column_buffers[0],
                                             qual_handle=column_buffers[1],
                                             meta_handle=column_buffers[2],
                                             num_records=num_reads)
        else:
            assert len(column_buffers) == 2
            return persona_ops.no_meta_bwa_assembler(bwa_read_pool=agd_read_pool,
                                                     base_handle=column_buffers[0],
                                                     qual_handle=column_buffers[1],
                                                     num_records=num_reads)

    if agd_read_pool is None:
        agd_read_pool = persona_ops.bwa_read_pool(**agd_read_pool_args, name="agd_reader_bwa_read_pool")

    assert len(upstream_tensors) > 0
    for output_buffers, num_reads in upstream_tensors:
        yield make_agd_read(column_buffers=output_buffers, num_reads=num_reads)

def agd_read_assembler(upstream_tensors, agd_read_pool=None, agd_read_pool_args=pool_default_args, include_meta=False, name="agd_read_assembler"):
    """
    Generate agd_read datatypes from the upstream tensors
    :param upstream_tensors: a list of tuples of tensors with type: (column_buffers, num_reads)
    :param agd_read_pool: if not None, pass in an instance of persona_ops.agd_read_pool to share
    :param agd_read_pool_args: args for deafult construction of agd_read_pool if it's None
    :param include_meta: create a meta read assembler if passed. The shape of upstream_tensors must be compatible
    :param name: 
    :return: yield instances of a tensor with AGDRead instance as the result
    """
    def make_agd_read(column_buffers, num_reads):
        # order on column_buffers: bases, quals, metadata (if exists)
        if isinstance(column_buffers, ops.Tensor):
            column_buffers = array_ops.unstack(column_buffers)
        if include_meta:
            assert len(column_buffers) == 3
            return persona_ops.agd_assembler(agd_read_pool=agd_read_pool,
                                             base_handle=column_buffers[0],
                                             qual_handle=column_buffers[1],
                                             meta_handle=column_buffers[2],
                                             num_records=num_reads)
        else:
            assert len(column_buffers) == 2
            return persona_ops.no_meta_agd_assembler(agd_read_pool=agd_read_pool,
                                                     base_handle=column_buffers[0],
                                                     qual_handle=column_buffers[1],
                                                     num_records=num_reads)

    if agd_read_pool is None:
        agd_read_pool = persona_ops.agd_read_pool(**agd_read_pool_args, name="agd_reader_agd_read_pool")

    assert len(upstream_tensors) > 0
    for output_buffers, num_reads in upstream_tensors:
        yield make_agd_read(column_buffers=output_buffers, num_reads=num_reads)

def join(upstream_tensors, parallel, capacity, multi=False, name="join"):
    if not isinstance(upstream_tensors, (tuple, list)):
        upstream_tensors = tuple(upstream_tensors)
        if multi:
            raise Exception("single example or generator given to multi join")
    if multi:
        return batch_join_pdq(tensor_list_list=upstream_tensors, batch_size=1,
                              num_dq_ops=parallel, capacity=capacity, name=name)
    else:
        return batch_pdq(tensor_list=upstream_tensors, batch_size=1,
                         capacity=capacity, num_dq_ops=parallel, name=name)

#### old stuff ####

def _parse_pipe(data_in, capacity, process_parallel, buffer_pool, name=None):

  to_enqueue = []
  for chunks_and_key in data_in:
    mapped_chunks = chunks_and_key[:-1]
    mapped_key = chunks_and_key[-1]
    parsed_chunks = []
    for mapped_chunk in mapped_chunks:
      if mapped_chunk.get_shape() == tensor_shape.TensorShape([2]):
          m_chunk = array_ops.expand_dims(mapped_chunk, 0)
      else:
          m_chunk = mapped_chunk

      parsed_chunk, num_records, first_ordinal = persona_ops.agd_reader(verify=False,
                              buffer_pool=buffer_pool, file_handle=m_chunk, name=name)
      parsed_chunk_u = array_ops.unstack(parsed_chunk)[0]

      parsed_chunks.append(parsed_chunk_u)

    # we just grab the last num recs and first ord because they should be
    # all the same ... i.e. columns are from the same group and share indices
    num_recs_u = array_ops.unstack(num_records)[0]
    first_ord_u = array_ops.unstack(first_ordinal)[0]
    parsed_chunks.insert(0, first_ord_u)
    parsed_chunks.insert(0, num_recs_u)
    parsed_chunks.insert(0, mapped_key)
    to_enqueue.append(parsed_chunks)

  parsed = batch_join_pdq(tensor_list_list=[e for e in to_enqueue],
                                        batch_size=1, capacity=capacity,
                                        enqueue_many=False,
                                        num_dq_ops=process_parallel,
                                        name=name)

  return parsed

def _key_maker(file_keys):
  num_file_keys = len(file_keys)

  string_producer = training.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
  sp_output = string_producer.dequeue()

  #keys = tf.train.batch_pdq([sp_output], batch_size=1, num_dq_ops=1, name="key_queue")

  #return keys[0]  # just the one
  return sp_output

def _keys_maker(file_keys, read_parallel):
  num_file_keys = len(file_keys)

  string_producer = training.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
  sp_output = string_producer.dequeue()

  keys = batch_pdq([sp_output], batch_size=1, num_dq_ops=read_parallel, name="keys_queue")

  return keys

"""
Build an input pipeline to get columns from an AGD dataset.

  ops = persona_in_pipe("dataset/metadata.json", ["base","qual"])
  
columns: list of extensions, which columns to read in, must be in same group
key: optional scalar tensor with chunk key (otherwise adds a string_input_producer for all
  chunks in `metadata_path`
returns: list of tensors in the form [ key, num_records, first_ordinal, col1, col1, ... , colN ]
where col0 - colN are Tensor([2]) and all else is scalar
"""
def persona_in_pipe(columns, dataset_dir, metadata_path=None, key=None, mmap_pool=None,
    buffer_pool=None, parse_parallel=2, process_parallel=1, sync=True, capacity=32, name=None):

  if metadata_path is not None:
    with open(metadata_path, 'r') as j:
      manifest = json.load(j)

    records = manifest['records']
    chunknames = []
    for record in records:
      chunknames.append(record['path'])

    print(dataset_dir)
    # verify that the desired columns exist
    for extension in columns:
      file_name = dataset_dir + "/" + chunknames[0] + "." + extension
      if not os.path.isfile(file_name):
        raise Exception("Desired column file {col} does not exist in AGD dataset {dataset}".format(col=file_name, dataset=metadata_path))


  with ops.name_scope(name, "persona_in_pipe", [key, mmap_pool, buffer_pool]):

    if mmap_pool is None:
      mmap_pool = persona_ops.m_map_pool(size=10, bound=False, name=name)
    if buffer_pool is None:
      buffer_pool = persona_ops.buffer_pool(size=10, bound=False, name=name)

    if key is None:
      # construct input producer
      if metadata_path is None:
        raise Exception("If keys is None, must also pass a valid metadata file")
      key = _key_maker(chunknames)

    chunk_filenames = []
    for extension in columns:
      extension_op = constant_op.constant(extension)
      new_name = string_ops.string_join([key, suffix_separator, extension_op]) # e.g. key.results
      chunk_filenames.append(new_name)

    # cascading MMAP operations give better disk performance
    chunks, names = persona_ops.file_m_map(filename=chunk_filenames[0], name=name, pool_handle=mmap_pool,
                                                  local_prefix=dataset_dir, synchronous=sync)

    all_chunks = []
    all_chunks.append(chunks)
    prev = chunks
    for chunk_filename in chunk_filenames[1:]:
      with ops.control_dependencies([prev]):
        chunks, names = persona_ops.file_m_map(filename=chunk_filename, name=name, pool_handle=mmap_pool,
                                                  local_prefix=dataset_dir, synchronous=sync)
      all_chunks.append(chunks)
      prev = chunks

    all_chunks.append(key)

    mmap_queue = batch_pdq(all_chunks, batch_size=1,
                                      enqueue_many=False,
                                      num_dq_ops=parse_parallel,
                                      name=name)

    to_enqueue = []

    return _parse_pipe(mmap_queue, capacity, process_parallel, buffer_pool, name)

"""
Build an input pipeline to get columns from an AGD dataset in a Ceph object store.
Expects Ceph keyring and config files to be in PWD.

  ops = persona_ceph_in_pipe("dataset/metadata.json", ["base","qual"])
  
columns: list of extensions, which columns to read in, must be in same group
keys: optional list of scalar tensors with chunk keys (otherwise adds a string_input_producer for all
  chunks in `metadata_path`)
ceph_params: dict of ceph parameters. ["cluster_name", "user_name", "ceph_conf_path"]
read_parallel: how many sets of `columns` to read in parallel, will increase memory usage.
parse_parallel: how many sets of `columns` to parse(decompress) in parallel
process_parallel: how many sets of parsed `columns` to return
returns: list of tensors in the form [ key, num_records, first_ordinal, col0, col1, ... , colN ]*`process_parallel`
where col0 - colN are Tensor([2]) and all else is scalar
"""
def persona_ceph_in_pipe(columns, ceph_params, metadata_path=None, keys=None,
                         buffer_pool=None, parse_parallel=2, read_parallel=1, process_parallel=1, ceph_read_size=2**26, capacity=32, name=None):

  if metadata_path is not None:
    with open(metadata_path, 'r') as j:
      manifest = json.load(j)

    records = manifest['records']
    chunknames = []
    for record in records:
      chunknames.append(record['path'])

  cluster_name = ceph_params["cluster_name"]
  user_name = ceph_params["user_name"]
  ceph_conf_path = ceph_params["ceph_conf_path"]
  pool_name = ceph_params["pool_name"]

  #TODO a way to check that chunk columns exist in the ceph store?

  with ops.name_scope(name, "persona_ceph_in_pipe", [keys, buffer_pool]):

    if buffer_pool is None:
      buffer_pool = persona_ops.buffer_pool(size=10, bound=False, name=name)

    if keys is None:
      if metadata_path is None:
        raise Exception("If keys is None, must also pass a valid metadata file")
      # construct input producer
      keys = _keys_maker(chunknames, read_parallel)

    suffix_sep = constant_op.constant(".")
    extension_ops = []
    for extension in columns:
      extension_ops.append(constant_op.constant(extension))

    chunk_buffers_list = []
    for key in keys:
      chunk_buffers = []
      chunk_filenames = []

      for i, extension in enumerate(columns):
        new_name = string_ops.string_join([key, suffix_sep, extension_ops[i]]) # e.g. key.results
        chunk_filenames.append(new_name)

      for chunk_filename in chunk_filenames:
        # [0] because ceph_reader also outputs filename which we don't need
        bb = persona_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name, pool_name=pool_name,
                                ceph_conf_path=ceph_conf_path, read_size=ceph_read_size, buffer_pool=buffer_pool,
                                queue_key=chunk_filename, name=name)[0]
        chunk_buffers.append(bb)
      chunk_buffers.append(key)
      chunk_buffers_list.append(chunk_buffers)

    chunk_queue = batch_join_pdq(chunk_buffers_list, batch_size=1,
                                      enqueue_many=False,
                                      num_dq_ops=parse_parallel,
                                      name=name)


    return _parse_pipe(chunk_queue, capacity, process_parallel, buffer_pool, name)

"""
Interpret pairs in a buffer_list in order of `columns` and write them out to disk beside metadata.
Uses a batch_join queue internally.

path: the for the dataset. will overwrite existing files with same keys and extension
columns: list of extensions of columns to write. must match in length to buffer_lists
write_list_list: list containing tuples (buffer_list, key, num_records, first_ordinal)
record_id: the ID to write into the column chunk headers
name: the op name for the pipeline
returns: list of tensor containing [key, num_records, first_ordinal]
"""
def persona_out_pipe(path, columns, write_list_list, record_id, compress=False, name=None):

  if path[-1] != '/':
    path.append('/')

  for item in write_list_list:
    if len(item) != 4:
      raise Exception("Expected items in write_list_list to be lists of len 4, got len {}".format(len(item)))
    if item[0].get_shape() != tensor_shape.TensorShape([2]):
      raise Exception("Expected shape of buffer_list to be [2], got {}".format(item[0].get_shape()))
    if item[1].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[1].get_shape()))
    if item[2].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[2].get_shape()))
    if item[3].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[3].get_shape()))

  with ops.name_scope(name, "persona_out_pipe", [write_list_list]):
    final_write_out = []
    for buff_list, key, num_records, first_ordinal in write_list_list:
      file_key_passthru, first_o_passthru = persona_ops.agd_write_columns(record_id=record_name,
                                                                    record_type=columns,
                                                                    column_handle=buff_list,
                                                                    compress=compress,
                                                                    output_dir=path,
                                                                    file_path=key,
                                                                    first_ordinal=first_ordinal,
                                                                    num_records=num_records,
                                                                    name=name)
      final_write_out.append([file_key_passthru, num_records, first_o_passthru])

    sink_queue = batch_join_pdq(final_write_out, capacity=10, num_dq_ops=1, batch_size=1, name=name)

    return sink_queue


"""
interpret pairs in a buffer_list as subchunks of a single column write it out to disk 
in `path`.

path: the for the dataset. will overwrite existing files with same keys and extension
columns: extension to write e.g. "results". If multiple buffer_lists, must be a list
write_list_list: list containing tuples (buffer_list(s), key, num_records, first_ordinal)
record_id: the ID to write into the column chunk headers
name: the op name for the pipeline
returns: tensor containing key
"""
def persona_parallel_out_pipe(path, column, write_list_list, record_id, compress=False, name=None):

  if path[-1] != '/':
    path += '/'

  if not isinstance(column, (list, tuple)):
    column = [column]

  for item in write_list_list:
    if len(item) != 4:
      raise Exception("Expected items in write_list_list to be lists of len 4, got len {}".format(len(item)))
    # item 0, the buffer_list, could be multiple buffer lists if there are secondary results
    if item[0].get_shape() != tensor_shape.TensorShape([2]):
      if item[0].get_shape().ndims == 2:

        if len(column) != item[0].get_shape().dims[0]:
          raise Exception("Expected number of columns supplied to be equal to number of buffer lists")
      else:
        raise Exception("Expected shape of buffer_list to be [2] or [N, 2], got {}".format(item[0].get_shape()))

    if item[1].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[1].get_shape()))
    if item[2].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[2].get_shape()))
    if item[3].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[3].get_shape()))

  with ops.name_scope(name, "persona_parallel_out_pipe", [write_list_list]):
    write_ops = []
    for buffer_list, key, num_records, first_ordinal in write_list_list:
      if buffer_list.get_shape().ndims == 2:
        bufs = array_ops.unstack(buffer_list)
      else:
        bufs = [buffer_list]
      writes = []
      for i, buf in enumerate(bufs):
          writer_op = persona_ops.parallel_column_writer(
              column_handle=buf,
              record_type=column[i],
              record_id=record_id,
              num_records=num_records,
              first_ordinal=first_ordinal,
              file_path=key, name=name,
              compress=compress, output_dir=path
          )
          writes.append(writer_op)
      write_ops.append(writes)

    sink_queue = batch_join_pdq(write_ops, capacity=10, num_dq_ops=1, batch_size=1, name=name)
    return sink_queue[0]

"""
Interpret pairs in a buffer_list in order of `columns` and write them out to ceph 
Uses a batch_join queue internally.

path: the for the dataset. will overwrite existing files with same keys and extension
columns: list of extensions of columns to write. must match in length to buffer_lists
write_list_list: list containing tuples (buffer_list, key, num_records, first_ordinal)
record_id: the ID to write into the column chunk headers
name: the op name for the pipeline
returns: list of tensor containing [key, num_records, first_ordinal]
"""
def persona_ceph_out_pipe(metadata_path, column, write_list_list, record_id, ceph_params, compress=False, name=None):

  with open(metadata_path, 'r') as j:
    manifest = json.load(j)

  cluster_name = ceph_params["cluster_name"]
  user_name = ceph_params["user_name"]
  ceph_conf = ceph_params["ceph_conf_path"]
  pool = manifest["pool"]

  for item in write_list_list:
    if len(item) != 4:
      raise Exception("Expected items in write_list_list to be lists of len 4, got len {}".format(len(item)))
    if item[0].get_shape() != tensor_shape.TensorShape([2]):
      raise Exception("Expected shape of buffer_list to be [2], got {}".format(item[0].get_shape()))
    if item[1].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[1].get_shape()))
    if item[2].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[2].get_shape()))
    if item[3].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[3].get_shape()))

  final_write_out = []
  for buff_list, key, num_records, first_ordinal in write_list_list:
    # the passthru probably isnt required ....
    file_key_passthru, first_o_passthru = persona_ops.agd_ceph_write_columns(cluster_name=cluster_name,
                                                                             user_name=user_name,
                                                                             pool_name=output_pool_name,
                                                                             ceph_conf_path=ceph_conf_path,
                                                                             record_id=record_id,
                                                                             record_type=column,
                                                                             column_handle=buff_list,
                                                                             file_path=key,
                                                                             first_ordinal=first_ordinal,
                                                                             num_records=num_records,
                                                                             name=name)
    final_write_out.append([file_key_passthru, num_records, first_o_passthru])

  sink_queue = batch_join_pdq(final_write_out, capacity=1, num_dq_ops=1, batch_size=1, name=name)
  return sink_queue[0]

"""
interpret pairs in a buffer_list as subchunks of a single column write it out to ceph 
in `path`.

path: the for the dataset. will overwrite existing files with same keys and extension
columns: extension to write e.g. "results". If multiple buffer_lists, must be a list
write_list_list: list containing tuples (buffer_list(s), key, num_records, first_ordinal)
record_id: the ID to write into the column chunk headers
name: the op name for the pipeline
returns: tensor containing key
"""
def persona_parallel_ceph_out_pipe(metadata_path, column, write_list_list, record_id, ceph_params, compress=False, name=None):

  with open(metadata_path, 'r') as j:
    manifest = json.load(j)

  cluster_name = ceph_params["cluster_name"]
  user_name = ceph_params["user_name"]
  ceph_conf = ceph_params["ceph_conf_path"]
  pool = manifest["pool"]

  if not isinstance(column, (list, tuple)):
    column = [column]

  for item in write_list_list:
    if len(item) != 4:
      raise Exception("Expected items in write_list_list to be lists of len 4, got len {}".format(len(item)))
    # item 0, the buffer_list, could be multiple buffer lists if there are secondary results
    if item[0].get_shape() != tensor_shape.TensorShape([2]):
      if item[0].get_shape().ndims == 2:

        if len(column) != item[0].get_shape().dims[0]:
          raise Exception("Expected number of columns supplied to be equal to number of buffer lists")
      else:
        raise Exception("Expected shape of buffer_list to be [2] or [N, 2], got {}".format(item[0].get_shape()))

    if item[1].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[1].get_shape()))
    if item[2].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[2].get_shape()))
    if item[3].get_shape() != tensor_shape.TensorShape([]):
      raise Exception("Expected shape of key to be [], got {}".format(item[3].get_shape()))

  with ops.name_scope(name, "persona_parallel_ceph_out_pipe", [write_list_list]):
    write_ops = []
    for buffer_list, key, num_records, first_ordinal in write_list_list:
      if buffer_list.get_shape().ndims == 2:
        bufs = array_ops.unstack(buffer_list)
      else:
        bufs = [buffer_list]
      for i, buf in enumerate(bufs):
          print("buf shape is: {}".format(buf.get_shape()))
          writer_op = persona_ops.ceph_writer(
              cluster_name=cluster_name,
              user_name=user_name,
              pool_name=pool,
              cepn_conf_path=ceph_conf,
              column_handle=buf,
              record_type=column[i],
              record_id=record_id,
              num_records=num_records,
              first_ordinal=first_ordinal,
              file_name=key, name=name,
              compress=compress,
          )
          write_ops.append(writer_op)

    sink_queue = batch_join_pdq([write_ops], capacity=10, num_dq_ops=1, batch_size=1, name=name)
    return sink_queue[0]

