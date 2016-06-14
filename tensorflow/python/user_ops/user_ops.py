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

def DenseReader(file_handle, pool_handle, batch_size, size_hint=None, name=None, verify=False):
  if size_hint:
    return gen_user_ops.dense_reader(pool_handle=pool_handle, file_handle=file_handle, verify=verify, batch_size=batch_size, size_hint=size_hint, name=name)
  return gen_user_ops.dense_reader(pool_handle=pool_handle, file_handle=file_handle, verify=verify, batch_size=batch_size, name=name)

# default is 2 for the shared resource ref
def _assert_matrix(shape, column_dim=2):
  if shape.ndims != 2:
    raise Exception("Expected a matrix shape from {shp}".format(shp=shape))
  if shape[1] != column_dim:
    raise Exception("Expected {exp} for shape[1], but got {actual}".format(
      exp=column_dim, actual=shape[1]))

_dread_str = "DenseReader"
ops.NoGradient(_dread_str)
@ops.RegisterShape(_dread_str)
def _DenseReaderShape(op):
  # just force the input to be a vector (will raise an exception if incorrect)
  handle_shape = op.inputs[0].get_shape()
  expected_handle_shape = tensor_shape.vector(2)
  if handle_shape != expected_handle_shape:
      raise Exception("dense reader requires handle shape {exp}, but got {actual}".format(
          exp=expected_handle_shape, actual=handle_shape))
  input_shape = op.inputs[1].get_shape()
  _assert_matrix(input_shape)
  batch_size = op.get_attr("batch_size")
  if batch_size < 1:
    raise Exception("dense reader expects a positive batch size. Received {}".format(batch_size))
  return [input_shape]

def FileMMap(queue, handle, name=None):
  return gen_user_ops.file_m_map(queue_handle=queue, pool_handle=handle, name=name)

def FileStorage(access_key, secret_key, host, bucket, queue, name=None):
  return gen_user_ops.file_storage(access_key=access_key, secret_key=secret_key, host=host,
                                   bucket=bucket, queue_handle=queue, name=name)

_fm_str = "FileMMap"
@ops.RegisterShape(_fm_str)
def _FileMMapShape(op):
  return [tensor_shape.matrix(rows=1,cols=2), tensor_shape.vector(1)]
ops.NoGradient(_fm_str)

_sink_str = "SinkOp"
def Sink(data, name=None):
  return gen_user_ops.sink(data=data, name=name)
ops.NoGradient(_sink_str)

@ops.RegisterShape(_sink_str)
def _SinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_matrix(data)
  return []

_sm_str = "StagedFileMap"
def StagedFileMap(queue, upstream_files, upstream_names, handle, name=None):
  return gen_user_ops.staged_file_map(queue_handle=queue, pool_handle=handle,
                                      upstream_refs=upstream_files,
                                      upstream_names=upstream_names, name=name)
ops.NoGradient(_sm_str)

@ops.RegisterShape(_sm_str)
def _StagedFileMapShape(op):
  upstream_files_shape = op.inputs[1].get_shape().dims
  upstream_names_shape = op.inputs[2].get_shape().dims
  upstream_files_shape[0] += 1
  upstream_names_shape[0] += 1
  return [upstream_files_shape, upstream_names_shape]

_dr_str = "DenseRecordCreator"
def DenseRecordCreator(bases, qualities, name=None):
  return gen_user_ops.dense_record_creator(bases=bases, qualities=qualities, name=name)
ops.NoGradient(_dr_str)

@ops.RegisterShape(_dr_str)
def _DenseRecordCreatorShape(op):
  base_shape = op.inputs[0].get_shape()
  qual_shape = op.inputs[1].get_shape()
  _assert_matrix(base_shape)
  _assert_matrix(qual_shape)
  if base_shape != qual_shape: # to compare number of rows
    raise Exception("Expected base shape ({b}) to be the same as qual shape ({q})".format(
      b=base_shape, q=qual_shape))
  return [base_shape]

_dram_str = "DenseRecordAddMetadata"
def DenseRecordAddMetadata(dense_records, metadata, name=None):
  return gen_user_ops.dense_record_add_metadata(dense_data=dense_records, metadata=metadata, name=name)
ops.NoGradient(_dram_str)

@ops.RegisterShape(_dram_str)
def _DenseRecordMetadataShape(op):
  dense_shapes = op.inputs[0].get_shape()
  metadata_shapes = op.inputs[1].get_shape()
  _assert_matrix(dense_shapes)
  _assert_matrix(metadata_shapes)
  if dense_shapes != metadata_shapes:
    raise Exception("Dense Shape is {dense}, not equal to metadata shape {md}".format(dense=dense_shapes, md=metadata_shapes))
  return [dense_shapes]

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

_pp_str = "ParserPool"
def ParserPool(size, size_hint=4194304):
    return gen_user_ops.parser_pool(size=size, size_hint=size_hint)

ops.NoGradient(_pp_str)
@ops.RegisterShape(_pp_str)
def _ParserPoolShape(op):
    return [tensor_shape.vector(2)]

_mmp_str = "MMapPool"
def MMapPool(size):
    return gen_user_ops.m_map_pool(size=size)
ops.NoGradient(_mmp_str)

# seems like there should be a better way to do this
@ops.RegisterShape(_mmp_str)
def _MMapPoolShape(op):
    return [tensor_shape.vector(2)]
