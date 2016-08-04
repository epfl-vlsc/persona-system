# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import os

# default is 2 for the shared resource ref
def _assert_matrix(shape, column_dim=2):
  if shape.ndims != 2:
    raise Exception("Expected a matrix shape from {shp}".format(shp=shape))
  if shape[1] != column_dim:
    raise Exception("Expected {exp} for shape[1], but got {actual}".format(
      exp=column_dim, actual=shape[1]))
  return shape[0]

def _assert_vec(shape, vec_length):
    if shape != tensor_shape.vector(vec_length):
        raise Exception("Expected vec({length}), but got {act}".format(length=vec_length, act=shape))

def _assert_scalar(shape):
    if shape != tensor_shape.scalar():
        raise Exception("expected scalar value from {act}".format(act=shape))

class FASTQReader(io_ops.ReaderBase):

    def __init__(self, batch_size, name=None):
        rr = gen_user_ops.fastq_reader(batch_size=batch_size,
                name=name)
        super(FASTQReader, self).__init__(rr)

ops.NoGradient("FASTQReader")
ops.RegisterShape("FASTQReader")(common_shapes.scalar_shape)

def FASTQDecoder(value, name=None):

    return gen_user_ops.decode_fastq(value, name=name)

ops.NoGradient("FASTQDecoder")

def DenseReader(file_handle, pool_handle, reserve=8192, name=None, verify=False):
  if reserve < 1:
    raise Exception ("Dense reader 'reserve' param must be strictly positive. Got {}".format(reserve))
  return gen_user_ops.dense_reader(buffer_pool=pool_handle, file_handle=file_handle, verify=verify, name=name, reserve=reserve)

_dread_str = "DenseReader"
ops.NoGradient(_dread_str)
@ops.RegisterShape(_dread_str)
def _DenseReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)
  input_shape = op.inputs[1].get_shape()
  _assert_matrix(input_shape)
  vec_shape = tensor_shape.vector(input_shape[0])
  a = [vec_shape] * 2
  return [input_shape] + a

def FileMMap(filename, handle, name=None):
  return gen_user_ops.file_m_map(filename=filename, pool_handle=handle, name=name)

_fm_str = "FileMMap"
@ops.RegisterShape(_fm_str)
def _FileMMapShape(op):
  filename_input = op.inputs[1].get_shape()
  pool_handle = op.inputs[0].get_shape()
  _assert_vec(pool_handle, 2)
  _assert_scalar(filename_input)
  return [tensor_shape.matrix(rows=1,cols=2), tensor_shape.vector(1)]
ops.NoGradient(_fm_str)

def S3Reader(access_key, secret_key, host, bucket, lookup_key, pool, name=None):
  return gen_user_ops.s3_reader(access_key=access_key, secret_key=secret_key, host=host,
                                bucket=bucket, key=lookup_key, pool_handle=pool, name=name)

_sr_str = "S3Reader"
@ops.RegisterShape(_sr_str)
def _S3ReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)

  key_shape = op.inputs[1].get_shape()
  _assert_scalar(key_shape)
  return [tensor_shape.vector(2), tensor_shape.vector(1)]
ops.NoGradient(_sr_str)

def CephReader(cluster_name, user_name, pool_name, ceph_conf_path, read_size, buffer_handle, queue_key, name=None):
  return gen_user_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name,
                                  pool_name=pool_name, ceph_conf_path=ceph_conf_path, read_size=read_size,
                                  buffer_handle=buffer_handle, queue_key=queue_key, name=name)

_cr_str = "CephReader"
@ops.RegisterShape(_cr_str)
def _CephReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)

  key_shape = op.inputs[1].get_shape()
  _assert_scalar(key_shape)
  return [tensor_shape.vector(2), tensor_shape.vector(1)]
ops.NoGradient(_cr_str)

def CephWriter(cluster_name, user_name, pool_name, ceph_conf_path, compress,
        record_id, record_type, column_handle, file_name, first_ordinal,
        num_records, name=None):
  return gen_user_ops.ceph_writer(cluster_name=cluster_name, user_name=user_name,
                                  pool_name=pool_name, ceph_conf_path=ceph_conf_path, compress=compress,
                                  record_id=record_id, record_type=record_type,
                                  column_handle=column_handle, file_name=file_name,
                                  first_ordinal=first_ordinal, num_records=num_records, name=name)

_cw_str = "CephWriter"
@ops.RegisterShape(_cw_str)
def _CephWriterShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)

  key_shape = op.inputs[1].get_shape()
  _assert_scalar(key_shape)
  return []
ops.NoGradient(_cw_str)

_read_sink_str = "ReadSink"
def ReadSink(data, name=None):
  return gen_user_ops.read_sink(data=data, name=name)
ops.NoGradient(_read_sink_str)

@ops.RegisterShape(_read_sink_str)
def _ReadSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return []

_buf_sink_str = "BufferSink"
def BufferSink(data, name=None):
  return gen_user_ops.buffer_sink(data=data, name=name)
ops.NoGradient(_buf_sink_str)

@ops.RegisterShape(_buf_sink_str)
def _BufferSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return []

_buf_list_sink_str = "BufferListSink"
def BufferListSink(data, name=None):
  return gen_user_ops.buffer_list_sink(data=data, name=name)
ops.NoGradient(_buf_list_sink_str)

@ops.RegisterShape(_buf_list_sink_str)
def _BufferListSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return []

_dt_string = "DenseTester"
def DenseTester(num_records, dense_records, genome_handle, sam_filename, name=None):
  if not (os.path.exists(sam_filename) and os.path.isfile(sam_filename)):
    raise EnvironmentError("DenseTester SAM file '{}' is not valid".format(sam_filename))
  return gen_user_ops.dense_tester(num_records=num_records, dense_records=dense_records,
                                   genome_handle=genome_handle, sam_filename=sam_filename, name=name)
ops.NoGradient(_dt_string)

@ops.RegisterShape(_dt_string)
def _DenseTesterShape(op):
  for i in range(2):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[2].get_shape())
#  return [tensor_shape.vector(2), tensor_shape.scalar()]
  return [op.inputs[1].get_shape(), op.inputs[2].get_shape()]

_sm_str = "StagedFileMap"
def StagedFileMap(filename, upstream_files, upstream_names, handle, name=None):
  return gen_user_ops.staged_file_map(filename=filename, pool_handle=handle,
                                      upstream_refs=upstream_files,
                                      upstream_names=upstream_names, name=name)
ops.NoGradient(_sm_str)

@ops.RegisterShape(_sm_str)
def _StagedFileMapShape(op):
  filename = op.inputs[0].get_shape()
  files = op.inputs[1].get_shape()
  names = op.inputs[2].get_shape()
  pool_handle = op.inputs[3].get_shape()
  _assert_vec(pool_handle, 2)
  _assert_scalar(filename)
  num_files = _assert_matrix(files)
  _assert_vec(names, num_files)
  files_shape = files.dims
  names_shape = names.dims
  files_shape[0] += 1
  names_shape[0] += 1
  return [files_shape, names_shape]

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

def GenomeIndex(filePath, name=None):

    return gen_user_ops.genome_index(genome_location=filePath, name=name);

@ops.RegisterShape("GenomeIndex")
def _SnapAlignDense(op):
    return [tensor_shape.vector(2)]

ops.NoGradient("GenomeIndex")

def AlignerOptions(cmdLine, name=None):

    return gen_user_ops.aligner_options(cmd_line=cmdLine, name=name);

ops.NoGradient("AlignerOptions")

def SnapAlign(genome, options, read, name=None):

    return gen_user_ops.snap_align(genome_handle=genome, options_handle=options, read=read, name=name)

ops.NoGradient("SnapAlign")

_sad_string = "SnapAlignDense"
def SnapAlignDense(genome, options, buffer_pool, read, name=None):

    return gen_user_ops.snap_align_dense(genome_handle=genome, options_handle=options,
            buffer_pool=buffer_pool, read=read, name=name)

ops.NoGradient(_sad_string)
@ops.RegisterShape(_sad_string)
def _SnapAlignDense(op):
    return [tensor_shape.vector(2)]

_sadp_string = "SnapAlignDenseParallel"
def SnapAlignDenseParallel(genome, options, buffer_list_pool, read, chunk_size, subchunk_size, threads, name=None):

    return gen_user_ops.snap_align_dense_parallel(genome_handle=genome, options_handle=options,
            buffer_list_pool=buffer_list_pool, read=read, chunk_size=chunk_size,
            subchunk_size=subchunk_size, threads=threads, name=name)

ops.NoGradient(_sadp_string)
@ops.RegisterShape(_sadp_string)
def _SnapAlignDenseParallel(op):
    return [tensor_shape.vector(2)]


_drp_str = "DenseReadPool"
def DenseReadPool(size=0, bound=False, name=None):
    return gen_user_ops.dense_read_pool(size=size, bound=bound, name=name)

ops.NoGradient(_drp_str)
@ops.RegisterShape(_drp_str)
def _DenseReadPoolShape(op):
    return [tensor_shape.vector(2)]

_mmp_str = "MMapPool"
def MMapPool(size=0, bound=False, name=None):
    return gen_user_ops.m_map_pool(size=size, bound=bound, name=name)

ops.NoGradient(_mmp_str)
@ops.RegisterShape(_mmp_str)
def _MMapPoolShape(op):
    return [tensor_shape.vector(2)]

_bp_str = "BufferPool"
def BufferPool(size, bound=True, name=None):
    return gen_user_ops.buffer_pool(size=size, bound=bound, name=name)

ops.NoGradient(_bp_str)
@ops.RegisterShape(_bp_str)
def _BufferPoolShape(op):
    return [tensor_shape.vector(2)]

_blp_str = "BufferListPool"
def BufferListPool(size, bound=True, name=None):
    return gen_user_ops.buffer_list_pool(size=size, bound=bound, name=name)

ops.NoGradient(_blp_str)
@ops.RegisterShape(_blp_str)
def _BufferListPoolShape(op):
    return [tensor_shape.vector(2)]

_cw_str = "ColumnWriter"
allowed_type_values = set(["base", "qual", "meta", "results"])
def ColumnWriter(column_handle, file_path, first_ordinal, num_records, record_id, record_type, compress=False, output_dir="", name=None):
    if record_type not in allowed_type_values:
        raise Exception("record_type ({given}) for ColumnWriter must be one of the following values: {expected}".format(
          given=record_type, expected=allowed_type_values))
    if output_dir != "" and output_dir[-1] != "/":
      output_dir += "/"
    return gen_user_ops.column_writer(
      column_handle=column_handle,
      file_path=file_path,
      record_type=record_type,
      first_ordinal=first_ordinal,
      num_records=num_records,
      compress=compress,
      record_id=record_id,
      output_dir=output_dir,
      name=name
    )

ops.NoGradient(_cw_str)
@ops.RegisterShape(_cw_str)
def _ColumnWriterShape(op):
  column_handle_shape = op.inputs[0].get_shape()
  _assert_vec(column_handle_shape, 2)
  for i in range(1,4):
    _assert_scalar(op.inputs[i].get_shape())
  return []

_pcw_str = "ParallelColumnWriter"
allowed_type_values = set(["base", "qual", "meta", "results"])
def ParallelColumnWriter(column_handle, file_path, first_ordinal, num_records, record_id, record_type, compress=False, output_dir="", name=None):
    if record_type not in allowed_type_values:
        raise Exception("record_type ({given}) for ColumnWriter must be one of the following values: {expected}".format(
          given=record_type, expected=allowed_type_values))
    if output_dir != "" and output_dir[-1] != "/":
      output_dir += "/"
    return gen_user_ops.parallel_column_writer(
      column_handle=column_handle,
      file_path=file_path,
      record_type=record_type,
      first_ordinal=first_ordinal,
      num_records=num_records,
      compress=compress,
      record_id=record_id,
      output_dir=output_dir,
      name=name
    )

ops.NoGradient(_pcw_str)
@ops.RegisterShape(_pcw_str)
def _ParallelColumnWriterShape(op):
  column_handle_shape = op.inputs[0].get_shape()
  _assert_vec(column_handle_shape, 2)
  for i in range(1,4):
    _assert_scalar(op.inputs[i].get_shape())
  return []

_da_str = "DenseAssembler"
def DenseAssembler(dense_read_pool, base_handle, qual_handle, meta_handle, num_records, name=None):
  return gen_user_ops.dense_assembler(
    dense_read_pool=dense_read_pool,
    base_handle=base_handle,
    qual_handle=qual_handle,
    meta_handle=meta_handle,
    num_records=num_records,
    name=name
  )

ops.NoGradient(_da_str)
@ops.RegisterShape(_da_str)
def _DenseAssemblerShape(op):
  # getting the input op
  _assert_vec(op.inputs[0].get_shape(), 2)
  for i in range(1,4):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[4].get_shape())
  return [tensor_shape.vector(2)]

_nmda_str = "NoMetaDenseAssembler"
def NoMetaDenseAssembler(dense_read_pool, base_handle, qual_handle, num_records, name=None):
  return gen_user_ops.no_meta_dense_assembler(
    dense_read_pool=dense_read_pool,
    base_handle=base_handle,
    qual_handle=qual_handle,
    num_records=num_records,
    name=name
  )

ops.NoGradient(_nmda_str)
@ops.RegisterShape(_nmda_str)
def _NoMetaDenseAssemblerShape(op):
  # getting the input op
  _assert_vec(op.inputs[0].get_shape(), 2)
  for i in range(1,3):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[3].get_shape())
  return [tensor_shape.vector(2)]

_dap_str = "DenseAssemblerPool"
def DenseAssemblerPool(size=0, bound=False, name=None):
    return gen_user_ops.dense_assembler_pool(size=size, bound=bound, name=name)

ops.NoGradient(_dap_str)
@ops.RegisterShape(_dap_str)
def _DenseAssemblerPoolShape(op):
    return [tensor_shape.vector(2)]

_fc_str = "FASTQCreator"
def FASTQCreator(data_handle, pool_handle, name=None):
    return gen_user_ops.fastq_creator(data_handle=data_handle,
                                      pool_handle=pool_handle,
                                      name=name)

ops.NoGradient(_fc_str)
@ops.RegisterShape(_fc_str)
def _FASTQCreatorOPShape(op):
    for i in range(2):
        a = op.inputs[i].get_shape()
        _assert_vec(a, 2)
    return [tensor_shape.vector(2)]

_fcp_str = _fc_str + "Pool"
def FASTQCreatorPool(size=0, bound=False, name=None):
    return gen_user_ops.fastq_creator_pool(size=size, bound=bound, name=name)

ops.NoGradient(_fcp_str)
@ops.RegisterShape(_fcp_str)
def _FASTQCreatorPoolOpShape(op):
    return [tensor_shape.vector(2)]

_gz_str = "GZIPDecomp"

def GZIPDecompressor(buffer_pool, data_handle, name=None):
  return gen_user_ops.gzip_decomp(buffer_pool=buffer_pool,
                                  data_handle=data_handle,
                                  name=name)

ops.NoGradient(_gz_str)
@ops.RegisterShape(_gz_str)
def _GZIPDecompShape(op):
  pool_shape = op.inputs[0].get_shape()
  data_shape = op.inputs[1].get_shape()
  _assert_vec(pool_shape, 2)
  _assert_vec(data_shape, 2)
  return [data_shape]
