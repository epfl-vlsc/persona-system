#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }

  using namespace std;
  using namespace errors;

  class AGDSortMetadataOp : public OpKernel {
  public:
    AGDSortMetadataOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    ~AGDSortMetadataOp() {
      core::ScopedUnref unref_listpool(bufferlist_pool_);
    }

    Status GetOutputBufferList(OpKernelContext* ctx, ResourceContainer<BufferList> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferlist_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("partial_handle", ctx));
      return Status::OK();
    }

    Status LoadDataResources(OpKernelContext* ctx, const Tensor* handles_t,
        vector<AGDRecordReader> &vec, const Tensor* num_records_t,
        vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>> &releasers) {
      auto rmgr = ctx->resource_manager();
      auto handles_matrix = handles_t->matrix<string>();
      auto num = handles_t->shape().dim_size(0);
      auto num_records = num_records_t->vec<int32>();
      ResourceContainer<Data> *input;

      for (int i = 0; i < num; i++) {
        TF_RETURN_IF_ERROR(rmgr->Lookup(handles_matrix(i, 0), handles_matrix(i, 1), &input));
        vec.push_back(AGDRecordReader(input, num_records(i)));
        releasers.push_back(move(vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>>::value_type(input, resource_releaser)));
      }
      return Status::OK();
    }
    

    void Compute(OpKernelContext* ctx) override {
      if (!bufferlist_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_list_pool", &bufferlist_pool_));
      }

      sort_index_.clear();

      const Tensor *results_in, *bases_in, *qualities_in, *metadata_in, *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->vec<int32>();
      OP_REQUIRES_OK(ctx, ctx->input("results_handles", &results_in));
      OP_REQUIRES_OK(ctx, ctx->input("bases_handles", &bases_in));
      OP_REQUIRES_OK(ctx, ctx->input("qualities_handles", &qualities_in));
      OP_REQUIRES_OK(ctx, ctx->input("metadata_handles", &metadata_in));
      
      vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>> releasers;

      int superchunk_records = 0;
      for (int i = 0; i < num_records.size(); i++)
        superchunk_records += num_records(i);

      Tensor* records_out_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("superchunk_records", TensorShape({}), &records_out_t));
      records_out_t->scalar<int32>()() = superchunk_records;

      vector<AGDRecordReader> metadata_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, metadata_in, metadata_vec, num_records_t, releasers));

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      //const format::AlignmentResult* agd_result;
      auto num_results = metadata_in->shape().dim_size(0);
      const char* data;
      size_t size;
      Status status;
      SortEntry entry;

      sort_index_.reserve(num_results * num_records(0));

      for (int i = 0; i < num_results; i++) {
        auto& metadata_reader = metadata_vec[i];
        status = metadata_reader.GetNextRecord(&data, &size);
        // go thru the results, build up vector of location, index, chunk
        int j = 0;
        while(status.ok()) {
          //agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          //entry.location = agd_result->location_;
          entry.meta = data;
          entry.meta_len = size;
          entry.chunk = i;
          entry.index = j;
          sort_index_.push_back(entry);
          /*if (entry.location < 0)
            LOG(INFO) << "location: " << entry.location << " at index " << entry.index << " in chunk "
              << entry.chunk << " appears to be invalid.";*/

          status = metadata_reader.GetNextRecord(&data, &size);
          j++;
        }
      }

      // phase 2: sort the vector by genome_location
      //LOG(INFO) << "running std sort on " << sort_index_.size() << " SortEntry's";
      std::sort(sort_index_.begin(), sort_index_.end(), [](const SortEntry& a, const SortEntry& b) {
          return strncmp(a.meta, b.meta, min(a.meta_len, b.meta_len)) < 0;
          });

      //LOG(INFO) << "Sort finished.";
      // phase 3: using the sort vector, merge the chunks into superchunks in sorted
      // order

      // now we need all the chunk data
      vector<AGDRecordReader> bases_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, bases_in, bases_vec, num_records_t, releasers));
      vector<AGDRecordReader> qualities_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, qualities_in, qualities_vec, num_records_t, releasers));
      vector<AGDRecordReader> results_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, results_in, results_vec, num_records_t, releasers));

      // get output buffer pairs (pair holds [index, data] to construct
      // AGD format temp output file in next dataflow stage)
      ResourceContainer<BufferList> *output_bufferlist_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferList(ctx, &output_bufferlist_container));
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

      for (size_t i = 0; i < sort_index_.size(); i++) {
        auto& entry = sort_index_[i];
        auto& result_reader = results_vec[entry.chunk];
        auto& bases_reader = bases_vec[entry.chunk];
        auto& qualities_reader = qualities_vec[entry.chunk];
        auto& metadata_reader = metadata_vec[entry.chunk];

        bases_reader.GetRecordAt(entry.index, &data, &size);
        bases_builder.AppendRecord(data, size);
        qualities_reader.GetRecordAt(entry.index, &data, &size);
        qualities_builder.AppendRecord(data, size);
        metadata_reader.GetRecordAt(entry.index, &data, &size);
        metadata_builder.AppendRecord(data, size);
        result_reader.GetRecordAt(entry.index, &data, &size);
        results_builder.AppendRecord(data, size);
      }

      // done
      LOG(INFO) << "DONE running sort by metadata!!";

    }

  private:
    ReferencePool<BufferList> *bufferlist_pool_ = nullptr;

    struct SortEntry {
      const char* meta;
      size_t meta_len;
      uint8_t chunk;
      int index;
    };

    vector<SortEntry> sort_index_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDSortMetadata").Device(DEVICE_CPU), AGDSortMetadataOp);
} //  namespace tensorflow {
