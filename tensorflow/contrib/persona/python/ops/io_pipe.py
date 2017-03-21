
# build input pipelines for AGD 


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.persona.python.ops.persona_ops import persona_ops as persona_ops_proxy
from tensorflow.contrib.persona.python.ops.queues import batch_pdq
from tensorflow.contrib.persona.python.ops.queues import batch_join_pdq
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import training

import json
import os

persona_ops = persona_ops_proxy()

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

    suffix_sep = constant_op.constant(".")
    chunk_filenames = []
    for extension in columns:
      extension_op = constant_op.constant(extension)
      new_name = string_ops.string_join([key, suffix_sep, extension_op]) # e.g. key.results
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
                                ceph_conf_path=ceph_conf_path, read_size=ceph_read_size, buffer_handle=buffer_pool,
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
      for i, buf in enumerate(bufs):
          print("buf shape is: {}".format(buf.get_shape()))
          writer_op = persona_ops.parallel_column_writer(
              column_handle=buf,
              record_type=column[i],
              record_id=record_id,
              num_records=num_records,
              first_ordinal=first_ordinal,
              file_path=key, name=name,
              compress=compress, output_dir=path
          )
          write_ops.append(writer_op)

    sink_queue = batch_join_pdq([write_ops], capacity=10, num_dq_ops=1, batch_size=1, name=name)
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

