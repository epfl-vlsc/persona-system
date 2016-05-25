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

def DenseReader(file_handle, batch_size, size_hint=None, name=None):
  if size_hint:
    return gen_user_ops.dense_reader(file_handle=file_handle, batch_size=batch_size, size_hint=size_hint, name=name) #, trace_file=trace_file)
  return gen_user_ops.dense_reader(file_handle=file_handle, batch_size=batch_size, name=name) #, trace_file=trace_file)

_dread_str = "DenseReader"
ops.NoGradient(_dread_str)
@ops.RegisterShape(_dread_str)
def _DenseReaderShape(op):
  # just force the input to be a vector (will raise an exception if incorrect)
  input_shape = op.inputs[0].get_shape()
  if input_shape != tensor_shape.vector(2):
    raise Exception("Got shape {actual}, but expected shape {exp}".format(
      actual=input_shape, exp=tensor_shape.vector(2)))
  batch_size = op.get_attr("batch_size")
  return [tensor_shape.scalar()]

def FileMMap(queue, name=None):
  return gen_user_ops.file_m_map(queue_handle=queue, name=name)

_fm_str = "FileMMap"
@ops.RegisterShape(_fm_str)
def _FileMMapShape(op):
  return [tensor_shape.matrix(rows=1,cols=2), tensor_shape.vector(1)]
ops.NoGradient(_fm_str)

_sm_str = "StagedFileMap"
def StagedFileMap(queue, upstream_files, upstream_names, name=None):
  return gen_user_ops.staged_file_map(queue_handle=queue,
                                      upstream_refs=upstream_files,
                                      upstream_names=upstream_names, name=name)
ops.NoGradient(_sm_str)

_dr_str = "DenseRecordCreator"
def DenseRecordCreator(bases, qualities, name=None):
  return gen_user_ops.dense_record_creator(bases=bases, qualities=qualities, name=name)
ops.NoGradient(_dr_str)

@ops.RegisterShape(_dr_str)
def _DenseRecordCreatorShape(op):
  base_shape = op.inputs[0].get_shape()
  qual_shape = op.inputs[1].get_shape()
  if base_shape != qual_shape:
    raise Exception("base shape is {base}, not equal to qual shape {qual}".format(
      base=base_shape, qual=qual_shape))
  return [base_shape]

_dram_str = "DenseRecordCreator"
def DenseRecordAddMetadata(dense_records, metadata, name=None):
  return gen_user_ops.dense_record_creator(dense_data=dense_records, metadata=metadata, name=name)
ops.NoGradient(_dram_str)

@ops.RegisterShape(_dram_str)
def _DenseRecordMetadataShape(op):
  dense_shapes = op.inputs[0].get_shape()
  metadata_shapes = op.inputs[1].get_shape()
  if dense_shapes != metadata_shapes:
    raise Exception("Dense Shape is {dense}, not equal to metadata shape {md}".format(dense=dense_shapes, md=metadata_shapes))
  return [dense_shapes]

@ops.RegisterShape(_sm_str)
def _StagedFileMapShape(op):
  upstream_files_shape = op.inputs[1].get_shape().dims
  upstream_names_shape = op.inputs[2].get_shape().dims
  upstream_files_shape[0] += 1
  upstream_names_shape[0] += 1
  return [upstream_files_shape, upstream_names_shape]

def DeleteColumn(input_tensor, name=None):
  return gen_user_ops.delete_column(data=input_tensor, name=name)
ops.NoGradient("DeleteColumn")

def Sink(input_tensor, name=None):
  return gen_user_ops.sink(data=input_tensor, name=name)
ops.NoGradient("Sink")

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
