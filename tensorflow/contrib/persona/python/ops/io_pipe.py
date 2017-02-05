
# build input pipelines for AGD 


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.persona.python.ops.persona_ops import persona_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import training

import json
import os

_persona_ops = persona_ops()

def _key_maker(file_keys):
  num_file_keys = len(file_keys)

  string_producer = training.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
  sp_output = string_producer.dequeue()

  #keys = tf.train.batch_pdq([sp_output], batch_size=1, num_dq_ops=1, name="key_queue")

  #return keys[0]  # just the one
  return sp_output

# columns: list of extensions, which columns to read in, must be in same group
# keys: optional scalar tensor with chunk keys (otherwise adds a string_input_producer)
# returns: list of tensors in the form [ key, num_records, first_ordinal, col0, col1, ... , colN ]
# where col0 - colN are Tensor([2]) and all else is scalar
def persona_in_pipe(metadata_path, columns, key=None, mmap_pool=None, 
    buffer_pool=None, parse_parallel=2, read_parallel=1, process_parallel=1, sync=True, capacity=32, name=None):
  
  with open(metadata_path, 'r') as j:
    manifest = json.load(j)

  records = manifest['records']
  chunknames = []
  for record in records:
    chunknames.append(record['path'])

  local_dir = os.path.dirname(metadata_path)
  print(local_dir)
  # verify that the desired columns exist
  for extension in columns:
    file_name = local_dir + "/" + chunknames[0] + "." + extension
    if not os.path.isfile(file_name):
      raise Exception("Desired column file {col} does not exist in AGD dataset {dataset}".format(col=file_name, dataset=metadata_path))
  

  with ops.name_scope(name, "persona_in_pipe", [key, mmap_pool, buffer_pool]):

    if mmap_pool is None:
      mmap_pool = _persona_ops.m_map_pool(size=10, bound=False, name=name)
    if buffer_pool is None:
      buffer_pool = _persona_ops.buffer_pool(size=10, bound=False, name=name)

    if key is None:
      # construct input producer
      key = _key_maker(chunknames)

    suffix_sep = constant_op.constant(".")
    chunk_filenames = []
    for extension in columns:
      extension_op = constant_op.constant(extension)
      new_name = string_ops.string_join([key, suffix_sep, extension_op]) # e.g. key.results
      chunk_filenames.append(new_name)

    # cascading MMAP operations give better disk performance
    chunks, names = _persona_ops.file_m_map(filename=chunk_filenames[0], name=name, pool_handle=mmap_pool,
                                                  local_prefix=local_dir, synchronous=sync)
    print("chunk shape after mmap is {}".format(chunks.get_shape()))
    for chunk_filename in chunk_filenames[1:]:
      if chunks.get_shape() == tensor_shape.TensorShape([2]):
        chunks = array_ops.expand_dims(chunks, 0)
      if names.get_shape() == tensor_shape.TensorShape([]):
        names = array_ops.expand_dims(names, 0)
      chunks, names = _persona_ops.staged_file_map(filename=chunk_filename, local_prefix=local_dir,
                                                  upstream_refs=chunks, upstream_names=names,
                                                  synchronous=sync, name=name, pool_handle=mmap_pool)
      print("chunk shape after mmap is {}".format(chunks.get_shape()))
 
    if chunks.get_shape() == tensor_shape.TensorShape([2]):
      all_chunks = [chunks]
    else:
      all_chunks = array_ops.unstack(chunks);
    all_chunks.append(key)
    
    mmap_queue = training.input.batch_pdq(all_chunks, batch_size=1,
                                      enqueue_many=False,
                                      num_dq_ops=parse_parallel,
                                      name=name)

    to_enqueue = []
    
    for chunks_and_key in mmap_queue:
      mapped_chunks = chunks_and_key[:-1]
      mapped_key = chunks_and_key[-1]
      parsed_chunks = []
      for mapped_chunk in mapped_chunks:
        print("mapped chunk shape is {}".format(mapped_chunk.get_shape()))
        if mapped_chunk.get_shape() == tensor_shape.TensorShape([2]):
            m_chunk = array_ops.expand_dims(mapped_chunk, 0)
        else:
            m_chunk = mapped_chunk
        
        print("mapped chunk shape is {}".format(mapped_chunk.get_shape()))
        
        parsed_chunk, num_records, first_ordinal = _persona_ops.agd_reader(verify=False, 
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

    parsed = training.input.batch_join_pdq(tensor_list_list=[e for e in to_enqueue],
                                          batch_size=1, capacity=capacity,
                                          enqueue_many=False,
                                          num_dq_ops=process_parallel,
                                          name=name)

    return parsed


# interpret pairs in buffer_list in order of columns and write them out to disk beside metadata
# returns: tensor containing 
# def persona_out_pipe(metadata_path, columns, buffer_list, write_parallel=1, compress_parallel=1) 



