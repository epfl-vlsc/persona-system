# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *

from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import io_ops

def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

class FASTQReader(io_ops.ReaderBase):

    def __init__(self, batch_size, name=None):
        rr = gen_user_ops.fastq_reader(batch_size=batch_size,
                name=name)
        super(FASTQReader, self).__init__(rr)

ops.NoGradient("FASTQReader")
ops.RegisterShape("FASTQReader")(common_shapes.scalar_shape)

def FASTQDecoder(value):

    return gen_user_ops.decode_fastq(value)

ops.NoGradient("FASTQDecoder")

def DenseReader(file_handle, batch_size):
  return gen_user_ops.dense_reader(file_handle=file_handle, batch_size=batch_size)

ops.NoGradient("DenseReader")
@ops.RegisterShape("DenseReader")
def _DenseReaderShape(op):
  # just force the input to be a vector (will raise an exception if incorrect)
  input_shape = op.inputs[0].get_shape().with_rank(1)
  batch_size = op.get_attr("batch_size")
  return [tensor_shape.TensorShape([batch_size]), tensor_shape.scalar()]

def DenseAggregator(bases, bases_count, qualities, qualities_count, metadata, metadata_count):
  return gen_user_ops.dense_aggregator(bases=bases,
                                       bases_count=bases_count,
                                       qualities=qualities,
                                       qualities_count=qualities_count,
                                       metadata=metadata,
                                       metadata_count=metadata_count)

@ops.RegisterShape("DenseAggregator")
def _DenseAggregatorShape(op): # pylint: disable=invalid-name
  # TODO this is such a hack
  bases = op.inputs[0]
  base_count = op.inputs[1]
  qualities = op.inputs[2]
  qualities_count = op.inputs[3]
  metadata = op.inputs[4]
  metadata_count = op.inputs[5]

  base_count_force = base_count.get_shape().with_rank(0)
  qualities_count_force = qualities_count.get_shape().with_rank(0)
  metadata_count_force = metadata_count.get_shape().with_rank(0)
  bases_force = bases.get_shape().with_rank(1)
  qualities_force = qualities.get_shape().with_rank(1)
  metadata_force = metadata.get_shape().with_rank(1)

  return [tensor_shape.TensorShape([3, bases_force[0]])]
ops.NoGradient("DenseAggregator")

def FileMMap(queue):
  return gen_user_ops.file_m_map(queue_handle=queue)

@ops.RegisterShape("FileMMap")
def _FileMMapSHape(op):
  return [tensor_shape.TensorShape([2])]
ops.NoGradient("FileMMap")

class SAMWriter(io_ops.WriterBase):

    def __init__(self, name=None, out_file=None):
        if out_file is None:
            out_file = name + '_out.txt'
        ww = gen_user_ops.sam_writer(name=name, out_file=out_file)
        super(SAMWriter, self).__init__(ww)

ops.NoGradient("SAMWriter")
ops.RegisterShape("SAMWriter")(common_shapes.scalar_shape)

class SAMAsyncWriter(io_ops.WriterBase):

    def __init__(self, name=None, out_file=None, num_buffers=16, buffer_size=1048576):
        if out_file is None:
            out_file = name + '_out.txt'
        ww = gen_user_ops.sam_async_writer(name=name, out_file=out_file, 
            num_buffers=num_buffers, buffer_size=buffer_size)
        super(SAMAsyncWriter, self).__init__(ww)

ops.NoGradient("SAMAsyncWriter")
ops.RegisterShape("SAMAsyncWriter")(common_shapes.scalar_shape)

def GenomeIndex(filePath):

    return gen_user_ops.genome_index(genome_location=filePath);

ops.NoGradient("GenomeIndex")

def AlignerOptions(cmdLine):

    return gen_user_ops.aligner_options(cmd_line=cmdLine);

ops.NoGradient("AlignerOptions")

def SnapAlign(genome, options, read):

    return gen_user_ops.snap_align(genome_handle=genome, options_handle=options, read=read)

ops.NoGradient("SnapAlign")
