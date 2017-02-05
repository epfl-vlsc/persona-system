
# build input pipelines for AGD 

# columns: list of extensions, which columns to read in, must be in same group
# names: optional scalar tensor with chunk keys (otherwise adds a string_input_producer)
# returns: list of tensors of shape 2 (resources) to chunks in order of columns
# def persona_in_pipe(metadata_path, columns, name=None, parse_parallel=2, read_parallel=1)

# interpret pairs in buffer_list in order of columns and write them out to disk beside metadata
# returns: tensor containing 
# def persona_out_pipe(metadata_path, columns, buffer_list, write_parallel=1, compress_parallel=1) 

from tensorflow.contrib.persona.python.ops.persona_ops import persona_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
