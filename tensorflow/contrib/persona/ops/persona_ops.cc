#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

using namespace errors;
using namespace std;
using namespace shape_inference;

// let the consumer write their own doc
#define REGISTER_REFERENCE_POOL(_NAME) \
  REGISTER_OP(_NAME) \
    .Attr("size: int") \
    .Attr("bound: bool = true") \
    .Attr("container: string = ''") \
    .Attr("shared_name: string = ''") \
    .SetShapeFn([](InferenceContext* c) { \
        c->set_output(0, c->Vector(2)); \
        return Status::OK(); \
        }) \
    .Output("pool_handle: Ref(string)") \
    .SetIsStateful()

  Status check_vector(InferenceContext *c, size_t input_idx, size_t dim_size) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 1, &input_data));
    auto dim_handle = c->Dim(input_data, 0);
    auto dim_value = c->Value(dim_handle);
    if (dim_value != dim_size) {
      return Internal("Op expected tensor of size ", dim_size, ", but got ", dim_value);
    }
    return Status::OK();
  }

  Status check_scalar(InferenceContext *c, size_t input_idx) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 0, &input_data));
    return Status::OK();
  }

  REGISTER_OP("AGDAssembler")
  .Input("agd_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("meta_handle: string")
  .Input("num_records: int32")
  .Output("agd_read_handle: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      for (int i = 0; i < 4; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
      }
      TF_RETURN_IF_ERROR(check_scalar(c, 4));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_OP("NoMetaAGDAssembler")
  .Input("agd_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("num_records: int32")
  .Output("agd_read_handle: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
      }
      TF_RETURN_IF_ERROR(check_scalar(c, 3));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_REFERENCE_POOL("AGDReadPool")
  .Doc(R"doc(
A pool specifically for agd read resources.

Intended to be used for AGDAssembler
)doc");

  REGISTER_OP("AGDCephMerge")
  .Attr("chunk_size: int >= 1")
  .Attr("intermediate_files: list(string)")
  .Attr("num_records: list(int)")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("file_buf_size: int = 10")
  .Input("buffer_list_pool: Ref(string)")
  .Output("chunk_out: string")
  .Output("num_recs: int32")
  .SetIsStateful()
  .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
num_records: vector of number of records
file_buf_size: the buffer size used for each individual file, default 10MB.
)doc");

  REGISTER_OP("AGDCephWriteColumns")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: list({'base', 'qual', 'metadata', 'results'})")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .Output("first_ordinal_out: int64")
  .SetIsStateful()
  .Doc(R"doc(
Writes out columns from a specified BufferList. The list contains
[data, index] BufferPairs. This Op constructs the header, unifies the buffers,
and writes to disk. Normally, this corresponds to a set of bases, qual, meta,
results columns.

This writes out to a Ceph object store only, defined by `cluster_name, user_name,
pool_name, and ceph_conf_path`.

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

)doc");

  REGISTER_OP("AGDConverter")
  .Input("buffer_list_pool: Ref(string)")
  .Input("input_data: string")
  .Output("agd_columns: string")
  .SetShapeFn([](InferenceContext *c) {
      for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
      }
      c->set_output(0, c->Vector(2));

      return Status::OK();
    })
  .Doc(R"doc(
Converts an input file into three files of bases, qualities, and metadata
)doc");

    REGISTER_OP("AGDMarkDuplicates")
  .Input("buffer_list_pool: Ref(string)")
  .Input("results_handle: string")
  .Input("num_records: int32")
  .Output("marked_results: string")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle input_data;
      for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));
        auto dim_handle = c->Dim(input_data, 0);
        auto dim_value = c->Value(dim_handle);
        if (dim_value != 2) {
          return Internal("AGDConverter input ", i, " must be a vector(2)");
        }
      }
      c->set_output(0, input_data);

      return Status::OK();
    })
  .SetIsStateful()
  .Doc(R"doc(
Mark duplicate reads/pairs that map to the same location.

This Op depends on data being sorted by metadata (QNAME),
i.e. A paired read is immediately followed by its mate.

Normally this step would be run on the aligner output before
sorting by genome location.

The implementation follows the approach used by SamBlaster
github.com/GregoryFaust/samblaster
wherein read pair signatures are looked up in a hash table
to determine if there are reads/pairs mapped to the exact
same location. Our implementation uses google::dense_hash_table,
trading memory for faster execution.
  )doc");

  REGISTER_OP("AGDMergeMetadata")
  .Attr("chunk_size: int >= 1")
  .Input("buffer_list_pool: Ref(string)")
  .Input("output_buffer_queue_handle: resource")
  .Input("chunk_group_handles: string") // a record of NUM_SUPER_CHUNKS x NUM_COLUMNS x 2 (2 for reference)
  .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`, using the metadata field
as sort key.

Op outputs a bufferlist with chunk columns in order: {meta, bases, quals, results}

Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
)doc");

  REGISTER_OP("AGDMerge")
  .Attr("chunk_size: int >= 1")
  .Input("buffer_list_pool: Ref(string)")
  .Input("output_buffer_queue_handle: resource")
  .Input("chunk_group_handles: string") // a record of NUM_SUPER_CHUNKS x NUM_COLUMNS x 2 (2 for reference)
  .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
*_handles: matrix of processed handles
output_buffer_queue_handle: a handle to a queue, into which are enqueued BufferList instance handles.
)doc");

  REGISTER_OP("AGDOutput")
  .Attr("unpack: bool = true")
  .Input("path: string")
  .Input("chunk_names: string")
  .Input("chunk_size: int32")
  .Input("start: int32")
  .Input("finish: int32")
  .SetIsStateful()
  .Doc(R"doc(
Takes a vector of string keys for AGD chunks, prefixed by `path`.

Prints records to stdout from record indices `start` to `finish`.

  )doc");

  REGISTER_OP("AGDReader")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("verify: bool = false")
  .Attr("reserve: int = 8192")
  .Attr("unpack: bool = true")
  .Attr("twobit: bool = false")
  .Input("buffer_pool: Ref(string)")
  .Input("file_handle: string")
  .Output("processed_buffers: string")
  .Output("num_records: int32")
  .Output("first_ordinal: int64")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &sh));
      auto dim_handle = c->Dim(sh, 0);
      auto dim_val = c->Value(dim_handle);

      c->set_output(0, c->input(1));
      c->set_output(1, c->Vector(dim_val));
      c->set_output(2, c->Vector(dim_val));

      return Status::OK();
    })
  .SetIsStateful()
  .Doc(R"doc(
Read in the agd format from an upstream source (file reader or network reader).

Outputs a handle to the buffer containing the processed data

Input buffer_pool is a handle to a tensorflow::BufferPoolOp result tensor,
and file_handle should come from a file_mmap_op

reserve: the number of bytes to call 'reserve' on the vector.
  )doc");

  REGISTER_OP("AGDSortMetadata")
  .Input("buffer_list_pool: Ref(string)")
  .Input("results_handles: string")
  .Input("bases_handles: string")
  .Input("qualities_handles: string")
  .Input("metadata_handles: string")
  .Input("num_records: int32")
  .Output("partial_handle: string")
  .Output("superchunk_records: int32")
  .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());

      return Status::OK();
    })
  .SetIsStateful()
  .Doc(R"doc(
Takes N results buffers, and associated bases, qualities and metadata
chunks, and sorts them into a merged a superchunk output buffer. This
is the main sort step in the AGD external merge sort.

This version sorts by metadata (QNAME in SAM).

Outputs handle to merged, sorted superchunks in `partial_handles`.
A BufferList that contains bases, qual, meta, results superchunk
BufferPairs ready for writing to disk.

Inputs -> (N, 2) string handles to buffers containing results, bases,
qualities and metadata. num_records is a vector of int32's with the
number of records per chunk.

Currently does not support a general number of columns.
The column order (for passing into AGDWriteColumns) is [bases, qualities, metadata, results]

  )doc");

  REGISTER_OP("AGDSort")
  .Input("buffer_list_pool: Ref(string)")
  .Input("results_handles: string")
  .Input("bases_handles: string")
  .Input("qualities_handles: string")
  .Input("metadata_handles: string")
  .Input("num_records: int32")
  .Output("partial_handle: string")
  .Output("superchunk_records: int32")
  .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());

      return Status::OK();
    })
  .SetIsStateful()
  .Doc(R"doc(
Takes N results buffers, and associated bases, qualities and metadata
chunks, and sorts them into a merged a superchunk output buffer. This
is the main sort step in the AGD external merge sort.

Outputs handle to merged, sorted superchunks in `partial_handles`.
A BufferList that contains bases, qual, meta, results superchunk
BufferPairs ready for writing to disk.

Inputs -> (N, 2) string handles to buffers containing results, bases,
qualities and metadata. num_records is a vector of int32's with the
number of records per chunk.

Currently does not support a general number of columns.
The column order (for passing into AGDWriteColumns) is [bases, qualities, metadata, results]

  )doc");

  REGISTER_OP("AGDVerifySort")
  .Input("path: string")
  .Input("chunk_names: string")
  .Input("chunk_size: int32")
  .SetIsStateful()
  .Doc(R"doc(
Verifies that the dataset referred to by `chunk_names` is sorted.

Chunk names must be in contiguous order.
  )doc");

  REGISTER_OP("AGDWriteColumns")
  .Attr("record_id: string")
  .Attr("record_type: list({'base', 'qual', 'metadata', 'results'})")
  .Attr("output_dir: string = ''")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .Output("first_ordinal_out: int64")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sh));
      auto dim_handle = c->Dim(sh, 0);
      auto dim_val = c->Value(dim_handle);
      if (dim_val != 2) {
        return Internal("column_handle must have dimensions {2}. Got ", dim_val);
      }

      for (int i = 1; i < 4; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &sh));
      }

      for (int i = 0; i < 2; i++) {
        c->set_output(i, c->input(1));
      }

      return Status::OK();
    })
  .Doc(R"doc(
Writes out columns from a specified BufferList. The list contains
[data, index] BufferPairs. This Op constructs the header, unifies the buffers,
and writes to disk. Normally, this corresponds to a set of bases, qual, meta,
results columns. Thus the BufferList corresponds to an entire dataset chunk.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

)doc");

  REGISTER_REFERENCE_POOL("BufferListPool")
  .Doc(R"doc(
Creates and initializes a pool containing a list of char buffers of size `buffer_size` bytes
  )doc");

  REGISTER_OP("BufferSink")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("data: string")
  .Doc(R"doc(
Consumes the buffer input and produces nothing
)doc");

  REGISTER_OP("BufferListSink")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("data: string")
  .Output("id: string")
  .Doc(R"doc(
Consumes the buffer input and produces nothing

Note that the output is meaningless. It's only purpose is so that
we can use it in other pipelines where writers are used
)doc");

  REGISTER_OP("CephReader")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("read_size: int")
  .Input("buffer_handle: Ref(string)")
  .Input("queue_key: string")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sh));
      auto dim_handle = c->Dim(sh, 0);
      auto dim_val = c->Value(dim_handle);
      if (dim_val != 2) {
        return Internal("buffer_handle must have dimensions {2}. Got ", dim_val);
      }
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &sh));
      for (int i = 0; i < 2; i++) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    })
  .Doc(R"doc(
Obtains file names from a queue, fetches those files from Ceph storage using Librados,
and writes them to a buffer from a pool of buffers.

buffer_handle: a handle to the buffer pool
queue_key: key reference to the filename queue
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  REGISTER_OP("CephWriter")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: {'base', 'qual', 'meta', 'results'}")
  .Input("column_handle: string")
  .Input("file_name: string")
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sh));
      auto dim_handle = c->Dim(sh, 0);
      auto dim_val = c->Value(dim_handle);
      if (dim_val != 2) {
        return Internal("buffer_handle must have dimensions {2}. Got ", dim_val);
      }
      for (int i = 1; i < 4; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &sh));
      }

      c->set_output(0, c->input(1)); // This op literally just copies the key

      return Status::OK();
    })
  .Doc(R"doc(
Writes data in column_handle to object file_name in specified Ceph cluster.

column_handle: a handle to the buffer pool
file_name: a Tensor() of string for the unique key for this file
compress: whether or not to compress the column
  )doc");

  REGISTER_OP("ColumnWriter")
  .Attr("record_id: string")
  .Attr("compressed: bool = false")
  .Attr("record_types: list({'base', 'qual', 'metadata', 'results'})")
  .Attr("output_dir: string = ''")
  .Input("columns: string")
  .Input("file_path: string")
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("file_path_out: string")
  .SetShapeFn([](InferenceContext *c) {
      vector<string> record_types;
      TF_RETURN_IF_ERROR(c->GetAttr("record_types", &record_types));

      ShapeHandle input_data;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_data));
      auto dim_handle = c->Dim(input_data, 1);
      auto dim_value = c->Value(dim_handle);
      if (dim_value != 2) {
        return Internal("columns must be an Nx2 matrix, but got ", dim_value, " for the 2nd dim");
      }

      dim_handle = c->Dim(input_data, 0);
      dim_value = c->Value(dim_handle);
      auto expected_size = record_types.size();
      if (dim_value != expected_size) {
        return Internal("columns must have ", expected_size, " in 0 dim, but got ", dim_value);
      }

      for (int i = 1; i < 4; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input_data));
      }

      c->set_output(0, c->input(1));

      return Status::OK();
    })
  .Doc(R"doc(
)doc");

  REGISTER_OP("FastqChunker")
  .Attr("chunk_size: int >= 1")
  .Input("queue_handle: resource")
  .Input("fastq_file: string") // TODO change this to resource when you update the op
  .Input("fastq_pool: Ref(string)")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle fastq_file;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &fastq_file));
      auto dim_handle = c->Dim(fastq_file, 0);
      auto fastq_dim = c->Value(dim_handle);
      if (fastq_dim != 2) {
        return Internal("fastq_file requires 2-dimensional vector");
      }

      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &fastq_file));
      dim_handle = c->Dim(fastq_file, 0);
      fastq_dim = c->Value(dim_handle);
      if (fastq_dim != 2) {
        return Internal("fastq_pool requires 2-dimensional vector");
      }

      return Status::OK();
    })
  .Doc(R"doc(

)doc");

  REGISTER_REFERENCE_POOL("FastqReadPool")
  .Doc(R"doc(
A pool to manage FastqReadResource objects
)doc");

  REGISTER_OP("FileMMap")
  .Attr("local_prefix: string = ''")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("synchronous: bool = false")
  .Input("pool_handle: Ref(string)")
  .Input("filename: string")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
      TF_RETURN_IF_ERROR(check_scalar(c, 1));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      return Status::OK();
      })
  .SetIsStateful()
  .Doc(R"doc(
Produces memory-mapped files, synchronously reads them, and produces a Tensor<2>
with the container and shared name for the file.

local_prefix: a directory on the local machine on which to find the keys
This is used in the case of a remote reader giving only the filenames to this reader
pool_handle: a handle to the filename queue
file_handle: a Tensor(2) of strings to access the shared mmaped file resource to downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  REGISTER_OP("StagedFileMap")
  .Attr("local_prefix: string = ''")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("synchronous: bool = false")
  .Input("filename: string")
  .Input("upstream_refs: string")
  .Input("upstream_names: string")
  .Input("pool_handle: Ref(string)")
  .Output("file_handles: string")
  .Output("file_names: string")
  .SetShapeFn([](InferenceContext *c) {
      ShapeHandle file_handles_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 2, &file_handles_shape));
      auto dim_handle = c->Dim(file_handles_shape, 0);
      auto dim_value = c->Value(dim_handle);

      c->set_output(0, c->Matrix(dim_value+1, 2));

      c->set_output(1, c->Vector(dim_value+1));

      return Status::OK();
    })
  .SetIsStateful()
  .Doc(R"doc(

  DEPRECATED


Appends a agd reader handle tensor to an input list.
To be used for the staged pipeline

local_prefix: a directory on the local machine on which to find the keys
This is used in the case of a remote reader giving only the filenames to this reader
pool_handle: handle to the filename queue
upstream_refs: the handles from previous stages in the pipeline, if any
upstream_names: the names from the previous stages of the pipeline, if any
file_handles: [{this file map op}] + upstream
file_names: [{this map op's name}] + upstream_name
)doc");

  REGISTER_REFERENCE_POOL("MMapPool")
  .Doc(R"doc(
Creates pools of MemoryMappedFile objects
)doc");

  REGISTER_OP("ParallelColumnWriter")
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: string")
  .Attr("output_dir: string = ''")
  .Attr("extension: string = ''")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
      })
  .SetIsStateful()
  .Doc(R"doc(
Writes out a column (just a character buffer) to the location specified by the input.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

This also assumes that this writer only writes out a single record type.
Thus we always need 3 of these for the full conversion pipeline
)doc");

  REGISTER_OP("ParallelSamWriter")
  .Attr("sam_file_path: string = ''")
  .Input("agd_results: string")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("read: string")
  .Input("num_records: int32")
  .Output("num_records_out: int32")
  .SetIsStateful()
  .Doc(R"doc(
Writes out the output in the SAM format (just a character buffer) to the location specified by the input.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);
)doc");

  REGISTER_OP("S3Reader")
  .Attr("access_key: string")
  .Attr("secret_key: string")
  .Attr("host: string")
  .Attr("bucket: string")
  .Input("pool_handle: Ref(string)")
  .Input("key: string")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetIsStateful()
  .Doc(R"doc(
Obtains file names from a queue, fetches those files from storage using S3, and writes
them to a buffer from a pool of buffers.

pool_handle: a handle to the filename queue
pool_handle: a handle to the buffer pool
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  REGISTER_OP("ZeroMqCSVSource")
  .Attr("url: string")
  .Attr("columns: int >= 1")
  .Output("output: string")
  .SetIsStateful()
  .Doc(R"doc(
  Creates a ZMQ reader that reads CSV line at a time from a ZMQ url of form tcp://blah:1234

  Op will pad or clip the CSV line to be exactly `columns` in terms of the length of `output`

  This dimension is specified by `columns`.
)doc");

  REGISTER_OP("ZeroMqSink")
  .Attr("url: string")
  .Input("input: string")
  .Doc(R"doc(
Creates a zmq writer that sends it's input to the specified URL
)doc");

  REGISTER_OP("ZeroMqSource")
  .Attr("url: string")
  .Output("output: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
  .Doc(R"doc(
  Creates a ZMQ reader that reads one line at a time from a ZMQ url of form tcp://blah:1234
)doc");

  REGISTER_REFERENCE_POOL("BufferPool")
  .Doc(R"doc(
Creates and initializes a pool containing char buffers of size `buffer_size` bytes
  )doc");

  REGISTER_OP("AGDTester")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("sam_filename: string = ''")
  .Input("genome_handle: Ref(string)")
  .Input("agd_records: string")
  .Input("num_records: int32")
  .Output("agd_records_out: string")
  .Output("num_records_out: int32")
  .Doc(R"doc(
  Compares the agd format output with the SAM format output
)doc");

#define MAKE_OP(_name_)                         \
  REGISTER_OP(_name_)                           \
  .Output("handle: Ref(string)")                \
  .Attr("cmd_line: string")                     \
  .Attr("container: string = ''")               \
  .Attr("shared_name: string = ''")             \
  .SetIsStateful()                              \
  .SetShapeFn([](InferenceContext *c) {         \
      c->set_output(0, c->Vector(2));           \
      return Status::OK();                      \
    })

MAKE_OP("AlignerOptions")
        .Doc(R"doc(
An op that produces SNAP aligner options.
handle: The handle to the options.
cmd_line: The SNAP command line parsed to create the options.
container: If non-empty, this options is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this options will be shared under the given name
  across multiple sessions.
)doc");

MAKE_OP("PairedAlignerOptions")
        .Doc(R"doc(
An op taht produces SNAP paired aligner options.
handle: The handle to the options.
cmd_line: The SNAP command line parsed to create the options.
container: If non-empty, this options is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this options will be shared under the given name
  across multiple sessions.
)doc");

  REGISTER_OP("GenomeIndex")
  .Output("handle: Ref(string)")
  .Attr("genome_location: string")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
  .Doc(R"doc(
    An op that creates or gives ref to a SNAP genome index.
    handle: The handle to the genomeindex resource.
    genome_location: The path to the genome index directory.
    container: If non-empty, this index is placed in the given container.
    Otherwise, a default container is used.
    shared_name: If non-empty, this queue will be shared under the given name
    across multiple sessions.
    )doc");

  REGISTER_OP("NullAligner")
  .Attr("subchunk_size: int")
  .Attr("wait_time_secs: float = 0.0")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->Matrix(1, 2));
      return Status::OK();
    })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
wait_time specifies the minimum time that the alignment should take
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

  REGISTER_OP("SnapAlignPaired")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_secondary: int >= 0")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      for (int i = 0; i < 4; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
      }
      int max_secondary = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));
      c->set_output(0, c->Matrix(1+max_secondary, 2));
      return Status::OK();
    })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.

Subchunk Size is the size in paired records. The actual chunk size will be 2x because of the pairing.
)doc");

  REGISTER_OP("SnapAlignSingle")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_secondary: int >= 0")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
      for (int i = 0; i < 4; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
      }
      int max_secondary = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));
      c->set_output(0, c->Matrix(1+max_secondary, 2));
      return Status::OK();
    })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

  REGISTER_OP("SnapIndexReferenceSequences")
    .Input("genome_handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
        })
      .Output("ref_seqs: string")
      .Output("ref_lens: int32")
      .SetIsStateful()
      .Doc(R"doc(
    Given a SNAP genome index, produce a string matrix containing the contigs
    (ref sequences).
    )doc");

  REGISTER_OP("BWAAlignSingle")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_read_size: int = 400")
  .Attr("max_secondary: int >= 1")
  .Input("index_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .SetShapeFn([](InferenceContext* c) {
      int max_secondary = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));

      c->set_output(0, c->Matrix(1+max_secondary, 2));
      return Status::OK();
      })
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
  Run single-ended alignment with BWA MEM. 
  max_secondary must be at least 1 for chimeric reads that BWA may output.
)doc");

  REGISTER_OP("BWAAligner")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_read_size: int = 400")
  .Input("index_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("read: string")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
  Using a number of threads, generates candidate alignments for
  the reads in `read`. Outputs the results in a BWACandidate
  object resource that should be passed to the BWAPairedEndStatOp node.
)doc");

  REGISTER_OP("BWAAssembler")
  .Input("bwa_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("meta_handle: string")
  .Input("num_records: int32")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("bwa_read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_OP("NoMetaBWAAssembler")
  .Input("bwa_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("num_records: int32")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("bwa_read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_REFERENCE_POOL("BWAReadPool")
  .Doc(R"doc(
A pool specifically for bwa read resources.

Intended to be used for BWAAssembler
)doc");

  REGISTER_OP("BWAFinalize")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_read_size: int = 400")
  .Attr("max_secondary: int >= 1")
  .Input("index_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .SetShapeFn([](InferenceContext* c) {
      int max_secondary = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));
      c->set_output(0, c->Matrix(1+max_secondary, 2));
      return Status::OK();
      })
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
  Complete the alignment process for candidates generated by BWAAligner.
  max_secondary must be at least 1 in case BWA outputs chimeric alignments of single reads.
)doc");

  REGISTER_OP("BWAIndex")
      .Output("handle: Ref(string)")
      .SetShapeFn([](InferenceContext* c) {
          c->set_output(0, c->Vector(2));
          return Status::OK();
          })
      .Attr("index_location: string")
      .Attr("ignore_alt: bool")
      .Attr("container: string = ''")
      .Attr("shared_name: string = ''")
      .SetIsStateful()
      .Doc(R"doc(
  An op that creates or gives ref to a bwa index.
  handle: The handle to the BWAIndex resource.
  genome_location: The path to the genome index directory.
  container: If non-empty, this index is placed in the given container.
  Otherwise, a default container is used.
  shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
  )doc");

  REGISTER_OP("BwaIndexReferenceSequences")
    .Input("index_handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
        })
    .Output("ref_seqs: string")
    .Output("ref_lens: int32")
    .SetIsStateful()
    .Doc(R"doc(
    Given a BWA genome index, produce two vectors containing the contigs
    (ref sequences) and their sizes.
    )doc");

  REGISTER_OP("BWAOptions")
      .Output("handle: Ref(string)")
      .Attr("options: list(string)")
      .Attr("container: string = ''")
      .Attr("shared_name: string = ''")
      .SetIsStateful()
      .Doc(R"doc(
  An op that creates or gives ref to a bwa index.
  handle: The handle to the BWAOptions resource.
  genome_location: The path to the genome index directory.
  container: If non-empty, this index is placed in the given container.
  Otherwise, a default container is used.
  shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
  )doc");

  REGISTER_OP("BWAPairedEndStat")
  .Input("index_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("read: string")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
  Using the mem_alnreg_v data generated by BWAAlignerOp,
  this op generates the pestat data, that being the insert size
  inferred from the read data in the chunk.

  This is the single threaded stage of processing a chunk.
)doc");

  REGISTER_OP("AgdOutputBam")
  .Attr("path: string")
  .Attr("pg_id: string")
  .Attr("ref_sequences: list(string)")
  .Attr("ref_seq_sizes: list(int)")
  .Attr("read_group: string")
  .Attr("sort_order: {'unknown', 'unsorted', 'queryname', 'coordinate'}")
  .Attr("num_threads: int >= 2")
  .Input("results_handle: string")
  .Input("bases_handle: string")
  .Input("qualities_handle: string")
  .Input("metadata_handle: string")
  .Input("num_records: int32")
  .SetShapeFn([](InferenceContext* c) {
    return NoOutputs(c);
    })
  .SetIsStateful()
  .Doc(R"doc(
  On execution, append reads/results chunks to output BAM file.

  Not all tags for SAM/BAM are currently supported, but support
  is planned. Currently supported is only required tags.

  RG and aux data is currently not supported but will be added soon.

  path: path for output .bam file
  pg_id: program id @PG for .bam
  ref_sequences: Reference sequences, @RG tags.
  ref_seq_sizes: Sizes of the references sequences.
  read_group: read group tag @RG
  *handles: the records to append to the BAM file
  num_records: the number of records held in *handles
  num_threads: number of threads for compression >= 2 because one is
  used to coordinate writing to disk.
  )doc");

}
