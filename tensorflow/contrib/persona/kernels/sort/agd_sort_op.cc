#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }

  using namespace std;
  using namespace errors;

  inline bool operator<(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() < rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() < rhs.position()) return true;
      else return false;
    } else
      return false;
  }

  class AGDSortOp : public OpKernel {
  public:
    AGDSortOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    ~AGDSortOp() {
      core::ScopedUnref unref_listpool(bufferpair_pool_);
    }

    Status GetOutputBufferPairs(OpKernelContext* ctx, vector<ResourceContainer<BufferPair>*>& ctrs)
    {
      Tensor* bufs_out_t;
      TF_RETURN_IF_ERROR(ctx->allocate_output("partial_handle", TensorShape({4, 2}), &bufs_out_t));
      auto bufs_out = bufs_out_t->matrix<string>();

      ctrs.resize(4);
      for (size_t i = 0; i < 4; i++) {

        TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(&ctrs[i]));
        ctrs[i]->get()->reset();
        bufs_out(i, 0) = ctrs[i]->container();
        bufs_out(i, 1) = ctrs[i]->name();
      }
      //TF_RETURN_IF_ERROR((*ctr)->allocate_output("partial_handle", ctx));
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
      if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));
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

      vector<AGDRecordReader> results_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, results_in, results_vec, num_records_t, releasers));

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      Alignment agd_result;
      auto num_results = results_in->shape().dim_size(0);
      const char* data;
      size_t size;
      Status status;
      SortEntry entry;

      sort_index_.reserve(num_results * num_records(0));

      for (int i = 0; i < num_results; i++) {
        auto& result_reader = results_vec[i];
        status = result_reader.GetNextRecord(&data, &size);
        // go thru the results, build up vector of location, index, chunk
        int j = 0;
        while(status.ok()) {
          //agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          bool check = agd_result.ParseFromArray(data, size);
          if (!check) {
            LOG(INFO) << "could not parse protobuf, data was ";
            fwrite(data, size, 1, stdout);
            LOG(INFO) << "and size was " << size;
          }
          entry.position = agd_result.position();
          entry.chunk = i;
          entry.index = j;
          sort_index_.push_back(entry);
          /*if (entry.location < 0)
            LOG(INFO) << "location: " << entry.location << " at index " << entry.index << " in chunk "
              << entry.chunk << " appears to be invalid.";*/

          status = result_reader.GetNextRecord(&data, &size);
          j++;
        }
      }

      // phase 2: sort the vector by genome_location
      LOG(INFO) << "running std sort on " << sort_index_.size() << " SortEntry's";
      std::sort(sort_index_.begin(), sort_index_.end(), [](const SortEntry& a, const SortEntry& b) {
          return a.position < b.position;
          });

      LOG(INFO) << "Sort finished.";
      // phase 3: using the sort vector, merge the chunks into superchunks in sorted
      // order

      // now we need all the chunk data
      vector<AGDRecordReader> bases_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, bases_in, bases_vec, num_records_t, releasers));
      vector<AGDRecordReader> qualities_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, qualities_in, qualities_vec, num_records_t, releasers));
      vector<AGDRecordReader> metadata_vec;
      OP_REQUIRES_OK(ctx, LoadDataResources(ctx, metadata_in, metadata_vec, num_records_t, releasers));

      // get output buffer pairs (pair holds [index, data] to construct
      // AGD format temp output file in next dataflow stage)
      vector<ResourceContainer<BufferPair>*> bufpair_containers;
      OP_REQUIRES_OK(ctx, GetOutputBufferPairs(ctx, bufpair_containers));
      
      ColumnBuilder bases_builder;
      ColumnBuilder qualities_builder;
      ColumnBuilder metadata_builder;
      ColumnBuilder results_builder;
      bases_builder.SetBufferPair(bufpair_containers[0]->get());
      qualities_builder.SetBufferPair(bufpair_containers[1]->get());
      metadata_builder.SetBufferPair(bufpair_containers[2]->get());
      results_builder.SetBufferPair(bufpair_containers[3]->get());

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
      LOG(INFO) << "DONE running sort!!";

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;

    struct SortEntry {
      // should double check actual size of this, using a protobuf object here might
      // not be a good idea
      Position position;
      uint8_t chunk;
      int index;
    };

    vector<SortEntry> sort_index_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDSort").Device(DEVICE_CPU), AGDSortOp);
} //  namespace tensorflow {
