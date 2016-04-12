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

from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import io_ops

def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

class FASTQReader(io_ops.ReaderBase):

    def __init__(self, name=None):
        rr = gen_user_ops.fastq_reader(name=name)
        super(FASTQReader, self).__init__(rr)

ops.NoGradient("FASTQReader")
ops.RegisterShape("FASTQReader")(common_shapes.scalar_shape)

def FASTQDecoder(value):

    return gen_user_ops.decode_fastq(value)

ops.NoGradient("FASTQDecoder")

class DenseReader(io_ops.ReaderBase):
    def __init__(self, name=None):
      import ipdb; ipdb.set_trace()
      rr = gen_user_ops.dense_reader(name=name)
      super(DenseReader, self).__init__(rr)
ops.NoGradient("DenseReader")
ops.RegisterShape("DenseReader")(common_shapes.scalar_shape)

class BaseReader(io_ops.ReaderBase):
    def __init__(self, name=None):
        rr = gen_user_ops.base_reader(name=name)
        super(BaseReader, self).__init__(rr)
ops.NoGradient("BaseReader")
ops.RegisterShape("BaseReader")(common_shapes.scalar_shape)

@ops.RegisterShape("FASTQDecoder")
def _FASTQDecoderShape(op):  # pylint: disable=invalid-name
  """Shape function for the FASTQDecoder op."""
  input_shape = op.inputs[0].get_shape()
  # Optionally check that all of other inputs are scalar or empty.
  for default_input in op.inputs[1:]:
    default_input_shape = default_input.get_shape().with_rank(1)
    if default_input_shape[0] > 1:
      raise ValueError(
          "Shape of a default must be a length-0 or length-1 vector.")
  return [input_shape] * len(op.outputs)

@ops.RegisterShape("DenseAggregator")
def _DenseAggregatorShape(op): # pylint: disable=invalid-name
  input_flat_shape = op.inputs[0].get_shape()
  for other in op.inputs[1:]:
    if other.get_shape() != input_flat_shape:
      raise ValueError("Flat shapes on DenseAggregators must be equal.")
  return [input_flat_shape]

class SAMWriter(io_ops.WriterBase):

    def __init__(self, name=None, out_file=None):
        if out_file is None:
            out_file = name + '_out.txt'
        ww = gen_user_ops.sam_writer(name=name, out_file=out_file)
        super(SAMWriter, self).__init__(ww)

ops.NoGradient("SAMWriter")
ops.RegisterShape("SAMWriter")(common_shapes.scalar_shape)

def GenomeIndex(filePath):

    return gen_user_ops.genome_index(genome_location=filePath);

ops.NoGradient("GenomeIndex")

def AlignerOptions(cmdLine):

    return gen_user_ops.aligner_options(cmd_line=cmdLine);

ops.NoGradient("AlignerOptions")

def SnapAlign(genome, options, read):

    return gen_user_ops.snap_align(genome_handle=genome, options_handle=options, read=read)

ops.NoGradient("SnapAlign")
