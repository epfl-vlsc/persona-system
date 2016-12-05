#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "agd_record_reader.h"
#include "compression.h"
#include "parser.h"
#include "util.h"
#include "buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {

  REGISTER_OP("AGDSort")
  .Input("buffer_pool: Ref(string)")
  .Input("results_handles: string")
  .Input("bases_handles: string")
  .Input("qualities_handles: string")
  .Input("metadata_handles: string")
  .Input("num_records: int32")
  .Output("partial_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Takes N results buffers, and associated bases, qualities and metadata
chunks, and sorts them into a merged a superchunk output buffer. This 
is the main sort step in the AGD external merge sort.

Outputs handle to merged, sorted superchunks in `partial_handles`. 
A BufferList that contains bases, qual, meta, results superchunk 
BufferPairs ready for writing to disk.

Inputs: (N, 2) string handles to buffers containing results, bases,
qualities and metadata. num_records is a vector of int32's with the 
number of records per chunk.

Currently does not support a general number of columns.

  )doc");

  using namespace std;
  using namespace errors;

  class AGDSortOp : public OpKernel {
  public:
    AGDSortOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }

    ~AGDSortOp() {
      core::ScopedUnref unref_pool(buffer_pool_);
    }
  
    Status GetOutputBufferList(OpKernelContext* ctx, ResourceContainer<BufferList> **ctr)
    {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("partial_handle", ctx));
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "bufferlist_pool", &bufferlist_pool_));
      }

      sort_index_.clear();

      const Tensor *results_in, bases_in, qualities_in, metadata_in, num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->vector<int32>();
      OP_REQUIRES_OK(ctx, ctx->input("results_handles", &results_in));
      OP_REQUIRES_OK(ctx, ctx->input("bases_handles", &bases_in));
      OP_REQUIRES_OK(ctx, ctx->input("qualities_handles", &qualities_in));
      OP_REQUIRES_OK(ctx, ctx->input("metadata_handles", &metadata_in));

      vector<AGDRecordReader> results_vec;
      LoadDataResources(ctx, results_in, results_vec, num_records_t);

      Tensor *output;
      TensorShape matrix_shape({2}); // basically a list of (container, name) pairs
      OP_REQUIRES_OK(ctx, ctx->allocate_output("processed_buffers", matrix_shape, &output));
      
      auto output_matrix = output->matrix<string>();

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      const format::AlignmentResult* agd_result;
      auto num_results = results_in->shape().dim_size(0);
      const char* data;
      uint32 size;
      Status status;
      SortEntry entry;

      for (int i = 0; i < num_results; i++) {
        auto& result_reader = results_vec[i];
        status = result_reader.GetNextRecord(&data, &size);
        // go thru the results, build up vector of location, index, chunk
        int j = 0;
        while(status.ok()) {
          agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          LOG(INFO) << "The result location is: " << agd_result->location_;
          entry.location = agd_result->location_;
          entry.chunk = i;
          entry.index = j;
          sort_index_.push_back(entry);

          status = result_reader.GetNextRecord(&data, &size);
          j++;
        }
      }

      // phase 2: sort the vector by genome_location
      std::qsort(sort_index_.begin(), sort_index_.end(), EntryCompare);

      // phase 3: using the sort vector, merge the chunks into superchunks in sorted
      // order

      // now we need all the chunk data
      vector<AGDRecordReader> bases_vec;
      LoadDataResources(ctx, bases_in, bases_vec, num_records_t);
      vector<AGDRecordReader> qualities_vec;
      LoadDataResources(ctx, qualities_in, qualities_vec, num_records_t);
      vector<AGDRecordReader> metadata_vec;
      LoadDataResources(ctx, metadata_in, metadata_vec, num_records_t);
       
      // get output buffer pairs (pair holds [index, data] to construct 
      // AGD format temp output file in next dataflow stage)
      ResourceContainer<BufferList> *output_bufferlist_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferList(ctx, &bufferlist_resource_container));
      auto output_bufferlist = output_bufferlist_container->get();
      output_bufferlist->resize(4);
      ColumnBuilder bases_builder;
      ColumnBuilder qualities_builder;
      ColumnBuilder metadata_builder;
      ColumnBuilder results_builder;
      bases_builder.SetBufferPair(&(*output_bufferlist)[0]);
      qualities_builder.SetBufferPair(&(*output_bufferlist)[1]);
      metadata_builder.SetBufferPair(&(*output_bufferlist)[2]);
      results_builder.SetBufferPair(&(*output_bufferlist)[3]);

      for (int i = 0; i < sort_index_.size(); i++) {
        auto& entry = sort_index_[i];
        auto& result_reader = results_vec[entry.chunk];
        auto& bases_reader = bases_vec[entry.chunk];
        auto& qualities_reader = qualities_vec[entry.chunk];
        auto& metdata_reader = metdata_vec[entry.chunk];

        bases_reader.GetRecordAt(&data, &size, entry.index);
        bases_builder.AppendRecord(data, size);
        qualities_reader.GetRecordAt(&data, &size, entry.index);
        qualities_builder.AppendRecord(data, size);
        metadata_reader.GetRecordAt(&data, &size, entry.index);
        metadata_builder.AppendRecord(data, size);
        results_reader.GetRecordAt(&data, &size, entry.index);
        results_builder.AppendRecord(data, size);
      }

      // done

    }

  private:
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
    ReferencePool<BufferList> *bufferlist_pool_ = nullptr;

    struct SortEntry {
      int64 location;
      int chunk;
      int index;
    };

    vector<SortEntry> sort_index_;

    bool EntryCompare(const SortEntry &a, const SortEntry &b) {
      return a.location < b.location; 
    }
    
    Status LoadDataResources(OpKernelContext* ctx, const Tensor* handles_t, 
        vector<AGDRecordReader> &vec, const Tensor* num_records_t) {
      auto rmgr = ctx->resource_manager();
      auto handles_matrix = handles_t->matrix<string>();
      auto num = handles_t->shape().dim_size(0);
      auto num_records = num_records_t->vector<int32>();
      ResourceContainer<Data> *input;

      for (int i = 0; i < num; i++) {
        OP_REQUIRES_OK(ctx, rmgr->Lookup(handles_matrix(i, 0), handles_matrix(i, 1), &input));
        vec.push_back(AGDRecordReader(input, num_records(i));
      }
    }

  };

  REGISTER_KERNEL_BUILDER(Name("AGDSort").Device(DEVICE_CPU), AGDSortOp);
} //  namespace tensorflow {
