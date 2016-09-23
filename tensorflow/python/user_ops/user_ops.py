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

from tensorflow.python.framework import ops, tensor_shape, common_shapes
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

_fq_str = "FASTQReader"
ops.NoGradient(_fq_str)
class FASTQReader(io_ops.ReaderBase):
    def __init__(self, batch_size, name=None):
        rr = gen_user_ops.fastq_reader(batch_size=batch_size,
                name=name)
        super(FASTQReader, self).__init__(rr)
ops.RegisterShape(_fq_str)(common_shapes.scalar_shape)

_ar_str = "AGDReader"
ops.NoGradient(_ar_str)
def AGDReader(file_handle, pool_handle, reserve=8192, name=None, verify=False):
  if reserve < 1:
    raise Exception ("AGD reader 'reserve' param must be strictly positive. Got {}".format(reserve))
  return gen_user_ops.agd_reader(buffer_pool=pool_handle, file_handle=file_handle, verify=verify, name=name, reserve=reserve)

@ops.RegisterShape(_ar_str)
def _AGDReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)
  input_shape = op.inputs[1].get_shape()
  _assert_matrix(input_shape)
  vec_shape = tensor_shape.vector(input_shape[0])
  a = [vec_shape] * 2
  return [input_shape] + a

_fm_str = "FileMMap"
ops.NoGradient(_fm_str)
def FileMMap(filename, handle, local_prefix="", name=None):
  if len(local_prefix) > 0 and not local_prefix.endswith("/"):
    local_prefix += "/"
  return gen_user_ops.file_m_map(filename=filename, pool_handle=handle, local_prefix=local_prefix, name=name)

@ops.RegisterShape(_fm_str)
def _FileMMapShape(op):
  filename_input = op.inputs[1].get_shape()
  pool_handle = op.inputs[0].get_shape()
  _assert_vec(pool_handle, 2)
  _assert_scalar(filename_input)
  return [tensor_shape.matrix(rows=1,cols=2), tensor_shape.vector(1)]

_sm_str = "StagedFileMap"
ops.NoGradient(_sm_str)
def StagedFileMap(filename, upstream_files, upstream_names, handle, local_prefix="", name=None):
  if len(local_prefix) > 0 and not local_prefix.endswith("/"):
    local_prefix += "/"
  return gen_user_ops.staged_file_map(filename=filename, pool_handle=handle,
                                      upstream_refs=upstream_files, local_prefix=local_prefix,
                                      upstream_names=upstream_names, name=name)

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

_sr_str = "S3Reader"
ops.NoGradient(_sr_str)
def S3Reader(access_key, secret_key, host, bucket, lookup_key, pool, name=None):
  return gen_user_ops.s3_reader(access_key=access_key, secret_key=secret_key, host=host,
                                bucket=bucket, key=lookup_key, pool_handle=pool, name=name)

@ops.RegisterShape(_sr_str)
def _S3ReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)

  key_shape = op.inputs[1].get_shape()
  _assert_scalar(key_shape)
  return [tensor_shape.vector(2), tensor_shape.vector(1)]

_cr_str = "CephReader"
ops.NoGradient(_cr_str)
def CephReader(cluster_name, user_name, pool_name, ceph_conf_path, read_size, buffer_handle, queue_key, name=None):
  return gen_user_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name,
                                  pool_name=pool_name, ceph_conf_path=ceph_conf_path, read_size=read_size,
                                  buffer_handle=buffer_handle, queue_key=queue_key, name=name)

@ops.RegisterShape(_cr_str)
def _CephReaderShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)

  key_shape = op.inputs[1].get_shape()
  _assert_scalar(key_shape)
  return [tensor_shape.vector(2), tensor_shape.vector(1)]

_cw_str = "CephWriter"
ops.NoGradient(_cw_str)
def CephWriter(cluster_name, user_name, pool_name, ceph_conf_path, compress,
        record_id, record_type, column_handle, file_name, first_ordinal,
        num_records, name=None):
  return gen_user_ops.ceph_writer(cluster_name=cluster_name, user_name=user_name,
                                  pool_name=pool_name, ceph_conf_path=ceph_conf_path, compress=compress,
                                  record_id=record_id, record_type=record_type,
                                  column_handle=column_handle, file_name=file_name,
                                  first_ordinal=first_ordinal, num_records=num_records, name=name)

@ops.RegisterShape(_cw_str)
def _CephWriterShape(op):
  handle_shape = op.inputs[0].get_shape()
  _assert_vec(handle_shape, 2)
  for i in range(1,4):
    scalar_shape = op.inputs[i].get_shape()
    _assert_scalar(scalar_shape)
  return [op.inputs[3].get_shape()]

_read_sink_str = "ReadSink"
ops.NoGradient(_read_sink_str)
def ReadSink(data, name=None):
  return gen_user_ops.read_sink(data=data, name=name)

@ops.RegisterShape(_read_sink_str)
def _ReadSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return []

_buf_sink_str = "BufferSink"
ops.NoGradient(_buf_sink_str)
def BufferSink(data, name=None):
  return gen_user_ops.buffer_sink(data=data, name=name)

@ops.RegisterShape(_buf_sink_str)
def _BufferSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return []

_buf_list_sink_str = "BufferListSink"
ops.NoGradient(_buf_list_sink_str)
def BufferListSink(data, name=None):
  return gen_user_ops.buffer_list_sink(data=data, name=name)

@ops.RegisterShape(_buf_list_sink_str)
def _BufferListSinkShape(op):
  data = op.inputs[0].get_shape()
  _assert_vec(data, 2)
  return [tensor_shape.scalar()]

_at_string = "AGDTester"
ops.NoGradient(_at_string)
def AGDTester(genome_handle, agd_records, num_records,  sam_filename, name=None):
  if not (os.path.exists(sam_filename) and os.path.isfile(sam_filename)):
    raise EnvironmentError("AGDTester SAM file '{}' is not valid".format(sam_filename))
  return gen_user_ops.agd_tester(num_records=num_records, agd_records=agd_records,
                                   genome_handle=genome_handle, sam_filename=sam_filename, name=name)

@ops.RegisterShape(_at_string)
def _AGDTesterShape(op):
  for i in range(2):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[2].get_shape())
#  return [tensor_shape.vector(2), tensor_shape.scalar()]
  return [op.inputs[1].get_shape(), op.inputs[2].get_shape()]

_sw_str = "SAMWriter"
ops.NoGradient("SAMWriter")
class SAMWriter(io_ops.WriterBase):
    def __init__(self, name=None, out_file=None):
        if out_file is None:
            out_file = name + '_out.txt'
        ww = gen_user_ops.sam_writer(name=name, out_file=out_file)
        super(SAMWriter, self).__init__(ww)
ops.RegisterShape("SAMWriter")(common_shapes.scalar_shape)

_saw_str = "SAMAsyncWriter"
ops.NoGradient("SAMAsyncWriter")
class SAMAsyncWriter(io_ops.WriterBase):
    def __init__(self, name=None, out_file=None, num_buffers=16, buffer_size=1048576):
        if out_file is None:
            out_file = name + '_out.txt'
        ww = gen_user_ops.sam_async_writer(name=name, out_file=out_file,
            num_buffers=num_buffers, buffer_size=buffer_size)
        super(SAMAsyncWriter, self).__init__(ww)
ops.RegisterShape("SAMAsyncWriter")(common_shapes.scalar_shape)

_gi_str = "GenomeIndex"
ops.NoGradient(_gi_str)
def GenomeIndex(index_path, name=None):
    return gen_user_ops.genome_index(genome_location=index_path, name=name);

@ops.RegisterShape(_gi_str)
def _GenomeIndexShape(op):
    return [tensor_shape.vector(2)]

_ao_str = "AlignerOptions"
ops.NoGradient(_ao_str)
def AlignerOptions(cmdLine, name=None):
    return gen_user_ops.aligner_options(cmd_line=cmdLine, name=name);

@ops.RegisterShape(_ao_str)
def _AlignerOptionsShape(op):
    return [tensor_shape.vector(2)]

_saap_string = "SnapAlignAGDParallel"
ops.NoGradient(_saap_string)
def SnapAlignAGDParallel(genome, options, buffer_list_pool, read, chunk_size, num_threads, subchunk_size, work_queue_size=10, sam_format=False, name=None):
  if num_threads < 1:
    raise EnvironmentError("number of threads must be greater than 0. Got {}".format(num_threads))
  return gen_user_ops.snap_align_agd_parallel(genome_handle=genome, options_handle=options, num_threads=num_threads,
                                              buffer_list_pool=buffer_list_pool, read=read, chunk_size=chunk_size,
                                              subchunk_size=subchunk_size, work_queue_size=work_queue_size, sam_format=sam_format, name=name)

@ops.RegisterShape(_saap_string)
def _SnapAlignAGDParallelShape(op):
    genome_handle = op.inputs[0].get_shape()
    options_handle = op.inputs[1].get_shape()
    buffer_list_pool = op.inputs[2].get_shape()
    read = op.inputs[3].get_shape()
    _assert_vec(genome_handle, 2)
    _assert_vec(options_handle, 2)
    _assert_vec(buffer_list_pool, 2)
    _assert_vec(read, 2)
    return [tensor_shape.vector(2)]

_na_string = "NullAligner"
ops.NoGradient(_na_string)
def NullAligner(buffer_list_pool, read, chunk_size, subchunk_size, extra_wait=0.0, name=None):
  return gen_user_ops.null_aligner(buffer_list_pool=buffer_list_pool, read=read, chunk_size=chunk_size,
                                              subchunk_size=subchunk_size, wait_time_secs=extra_wait, name=name)

@ops.RegisterShape(_na_string)
def _NullAligner(op):
    buffer_list_pool = op.inputs[0].get_shape()
    read = op.inputs[1].get_shape()
    _assert_vec(buffer_list_pool, 2)
    _assert_vec(read, 2)
    return [tensor_shape.vector(2)]

_arp_str = "AGDReadPool"
ops.NoGradient(_arp_str)
def AGDReadPool(size=0, bound=False, name=None):
    return gen_user_ops.agd_read_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_arp_str)
def _AGDReadPoolShape(op):
    return [tensor_shape.vector(2)]

_mmp_str = "MMapPool"
ops.NoGradient(_mmp_str)
def MMapPool(size=0, bound=False, name=None):
    return gen_user_ops.m_map_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_mmp_str)
def _MMapPoolShape(op):
    return [tensor_shape.vector(2)]

_bp_str = "BufferPool"
ops.NoGradient(_bp_str)
def BufferPool(size, bound=True, name=None):
    return gen_user_ops.buffer_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_bp_str)
def _BufferPoolShape(op):
    return [tensor_shape.vector(2)]

_blp_str = "BufferListPool"
ops.NoGradient(_blp_str)
def BufferListPool(size, bound=True, name=None):
    return gen_user_ops.buffer_list_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_blp_str)
def _BufferListPoolShape(op):
    return [tensor_shape.vector(2)]

_cw_str = "ColumnWriter"
ops.NoGradient(_cw_str)
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

@ops.RegisterShape(_cw_str)
def _ColumnWriterShape(op):
  column_handle_shape = op.inputs[0].get_shape()
  _assert_vec(column_handle_shape, 2)
  for i in range(1,4):
    _assert_scalar(op.inputs[i].get_shape())
  return []

_pcw_str = "ParallelColumnWriter"
ops.NoGradient(_pcw_str)
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

@ops.RegisterShape(_pcw_str)
def _ParallelColumnWriterShape(op):
  column_handle_shape = op.inputs[0].get_shape()
  _assert_vec(column_handle_shape, 2)
  for i in range(1,4):
    _assert_scalar(op.inputs[i].get_shape())
  return [op.inputs[3].get_shape()]

_psw_str = "ParallelSamWriter"
ops.NoGradient(_psw_str)
def ParallelSamWriter(agd_results, genome_handle, options_handle, read, num_records, sam_file_path, name=None):
      return gen_user_ops.parallel_sam_writer(
      agd_results = agd_results,
      genome_handle = genome_handle,
      options_handle = options_handle,
      read = read,
      num_records=num_records,
      sam_file_path=sam_file_path,
      name=name
    )

@ops.RegisterShape(_psw_str)
def _ParallelSamWriterShape(op):
  for i in range(0, 3):
    vec_shape = op.inputs[i].get_shape()
    _assert_vec(vec_shape, 2)
  num_records_shape = op.inputs[4].get_shape()
  _assert_scalar(num_records_shape)
  return [op.inputs[4].get_shape()]

_aa_str = "AGDAssembler"
ops.NoGradient(_aa_str)
def AGDAssembler(agd_read_pool, base_handle, qual_handle, meta_handle, num_records, name=None):
  return gen_user_ops.agd_assembler(
    agd_read_pool=agd_read_pool,
    base_handle=base_handle,
    qual_handle=qual_handle,
    meta_handle=meta_handle,
    num_records=num_records,
    name=name
  )

@ops.RegisterShape(_aa_str)
def _AGDAssemblerShape(op):
  # getting the input op
  _assert_vec(op.inputs[0].get_shape(), 2)
  for i in range(1,4):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[4].get_shape())
  return [tensor_shape.vector(2)]

_nmaa_str = "NoMetaAGDAssembler"
ops.NoGradient(_nmaa_str)
def NoMetaAGDAssembler(agd_read_pool, base_handle, qual_handle, num_records, name=None):
  return gen_user_ops.no_meta_agd_assembler(
    agd_read_pool=agd_read_pool,
    base_handle=base_handle,
    qual_handle=qual_handle,
    num_records=num_records,
    name=name
  )

@ops.RegisterShape(_nmaa_str)
def _NoMetaAGDAssemblerShape(op):
  # getting the input op
  _assert_vec(op.inputs[0].get_shape(), 2)
  for i in range(1,3):
    op_shape = op.inputs[i].get_shape()
    _assert_vec(op_shape, 2)
  _assert_scalar(op.inputs[3].get_shape())
  return [tensor_shape.vector(2)]

_aap_str = "AGDAssemblerPool"
ops.NoGradient(_aap_str)
def AGDAssemblerPool(size=0, bound=False, name=None):
    return gen_user_ops.agd_assembler_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_aap_str)
def _AGDAssemblerPoolShape(op):
    return [tensor_shape.vector(2)]

_fc_str = "FASTQCreator"
ops.NoGradient(_fc_str)
def FASTQCreator(data_handle, pool_handle, name=None):
    return gen_user_ops.fastq_creator(data_handle=data_handle,
                                      pool_handle=pool_handle,
                                      name=name)

@ops.RegisterShape(_fc_str)
def _FASTQCreatorOPShape(op):
    for i in range(2):
        a = op.inputs[i].get_shape()
        _assert_vec(a, 2)
    return [tensor_shape.vector(2)]

_fcp_str = _fc_str + "Pool"
ops.NoGradient(_fcp_str)
def FASTQCreatorPool(size=0, bound=False, name=None):
    return gen_user_ops.fastq_creator_pool(size=size, bound=bound, name=name)

@ops.RegisterShape(_fcp_str)
def _FASTQCreatorPoolOpShape(op):
    return [tensor_shape.vector(2)]

_gz_str = "GZIPDecomp"
ops.NoGradient(_gz_str)
def GZIPDecompressor(buffer_pool, data_handle, name=None):
  return gen_user_ops.gzip_decomp(buffer_pool=buffer_pool,
                                  data_handle=data_handle,
                                  name=name)

@ops.RegisterShape(_gz_str)
def _GZIPDecompShape(op):
  pool_shape = op.inputs[0].get_shape()
  data_shape = op.inputs[1].get_shape()
  _assert_vec(pool_shape, 2)
  _assert_vec(data_shape, 2)
  return [data_shape]

_ps_str = "PipeSource"
ops.NoGradient(_ps_str)
def PipeSource(path, create=False, name=None):
  return gen_user_ops.pipe_source(path=path, create=create)

@ops.RegisterShape(_ps_str)
def _PipeSourceShape(op):
  return [tensor_shape.scalar()]

_zmq_str = "ZeroMqSource"
ops.NoGradient(_zmq_str)
def ZeroMqSource(url, name=None):
  return gen_user_ops.zero_mq_source(url=url)

@ops.RegisterShape(_zmq_str)
def _ZeroMqPipeSourceShape(op):
  return [tensor_shape.scalar()]
