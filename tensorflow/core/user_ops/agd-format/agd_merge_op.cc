#include <vector>
#include <memory>
#include <utility>
#include <queue>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"

#include "format.h"
#include "compression.h"
#include "parser.h"
#include "util.h"
#include "buffer.h"
#include "agd_record_reader.h"

#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  namespace {
    const string op_name("AGDMerge");

    void resource_releaser(ResourceContainer<Data> *data) {
      core::ScopedUnref a(data);
      data->release();
    }

    class ColumnCursor {
    public:
      ColumnCursor(AGDRecordReader &&results, vector<AGDRecordReader> &&other_columns) :
        results_(move(results)), other_columns_(move(other_columns)) {}

      Status set_current_location() {
        const char* data;
        size_t data_sz;
        TF_RETURN_IF_ERROR(results_.PeekNextRecord(&data, &data_sz));
        current_result_ = reinterpret_cast<decltype(current_result_)>(data);
        current_location_ = current_result_->location_;
        return Status::OK();
      }

      Status append_to_buffer_list(BufferList *bl) {
        const char* data;

        // first, dump the alignment result in the first column
        auto &bp_results = (*bl)[0];
        TF_RETURN_IF_ERROR(copy_record(bp_results, results_));

        size_t bl_idx = 1;
        for (auto &r : other_columns_) {
          auto &bp = (*bl)[bl_idx++];
          TF_RETURN_IF_ERROR(copy_record(bp, r));
        }

        return Status::OK();
      }

      inline int64_t get_location() {
        return current_location_;
      }

    private:
      vector<AGDRecordReader> other_columns_;
      AGDRecordReader results_;
      const AlignmentResult *current_result_ = nullptr;
      int64_t current_location_ = -2048;

      static inline
      Status
      copy_record(BufferPair& bp, AGDRecordReader &r) {
        const char *record_data;
        size_t record_size;
        auto &index = bp.index();
        auto &data = bp.data();

        TF_RETURN_IF_ERROR(r.GetNextRecord(&record_data, &record_size));
        auto char_sz = static_cast<char>(record_size);
        TF_RETURN_IF_ERROR(index.AppendBuffer(&char_sz, sizeof(char_sz)));
        TF_RETURN_IF_ERROR(data.AppendBuffer(record_data, record_size));

        return Status::OK();
      }
    };

    typedef pair<int64_t, ColumnCursor*> GenomeScore;
    struct ScoreComparator {
      bool operator()(const GenomeScore &a, const GenomeScore &b) {
        return a.first > b.first;
      }
    };

  }

  REGISTER_OP(op_name.c_str())
  .Attr("chunk_size: int >= 1")
  .Attr("intermediate_files: list(string)")
  .Attr("path: string")
  .Attr("num_records: list(int)")
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
)doc");

  class AGDMergeOp : public OpKernel {
  public:
    AGDMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("intermediate_files", &intermediate_files_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_records", &num_records_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &path_));

      for (int i = 0; i < intermediate_files_.size(); i++) {
        string bases_file = path_ + "/" + intermediate_files_[i] + ".base";
        string meta_file = path_ + "/" + intermediate_files_[i] + ".metadata";
        string qual_file = path_ + "/" + intermediate_files_[i] + ".qual";
        string results_file = path_ + "/" + intermediate_files_[i] + ".results";

        unique_ptr<ReadOnlyMemoryRegion> bases_mmap;
        unique_ptr<ReadOnlyMemoryRegion> meta_mmap;
        unique_ptr<ReadOnlyMemoryRegion> qual_mmap;
        unique_ptr<ReadOnlyMemoryRegion> results_mmap;
        OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(bases_file, &bases_mmap));
        OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(meta_file, &meta_mmap));
        OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(qual_file, &qual_mmap));
        OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(results_file, &results_mmap));
        // the system is assuming the files are uncompressed and formatted with the usual header
        AGDRecordReader results_column((const char*)results_mmap->data() + sizeof(format::FileHeader), num_records_[i]);
        vector<AGDRecordReader> other_columns;
        other_columns.push_back(AGDRecordReader((const char*)bases_mmap->data() + sizeof(format::FileHeader), num_records_[i]));
        other_columns.push_back(AGDRecordReader((const char*)qual_mmap->data() + sizeof(format::FileHeader), num_records_[i]));
        other_columns.push_back(AGDRecordReader((const char*)meta_mmap->data() + sizeof(format::FileHeader), num_records_[i]));
        mapped_files_.push_back(move(results_mmap));
        mapped_files_.push_back(move(bases_mmap));
        mapped_files_.push_back(move(qual_mmap));
        mapped_files_.push_back(move(meta_mmap));

        
        ColumnCursor a(move(results_column), move(other_columns));
        OP_REQUIRES_OK(ctx, a.set_current_location());
        columns_.push_back(move(a));
      }
      
      for (auto &cc : columns_) {
        score_heap_.push(GenomeScore(cc.get_location(), &cc));
      }

    }

    ~AGDMergeOp() {

    }

    void Compute(OpKernelContext* ctx) override {
      if (!buflist_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      int current_chunk_size = 0;
      ColumnCursor *cc;
      ResourceContainer<BufferList> *bl_ctr;
      OP_REQUIRES_OK(ctx, buflist_pool_->GetResource(&bl_ctr));
      auto bl = bl_ctr->get();
      bl->resize(4);
      Status s;
      while (!score_heap_.empty()) {
        auto &top = score_heap_.top();
        cc = top.second;

        LOG(INFO) << "processing location: " << top.first;
        cc->append_to_buffer_list(bl);

        score_heap_.pop();

        s = cc->set_current_location();
        if (s.ok()) {
          // get_location will have the location advanced by the append_to_buffer_list call above
          score_heap_.push(GenomeScore(cc->get_location(), cc));
        } else if (!IsResourceExhausted(s)) {
          OP_REQUIRES_OK(ctx, s);
        } 

        // pre-increment because we just added 1 to the chunk size
        // we're guaranteed that chunk size is at least 1
        if (++current_chunk_size == chunk_size_) {
          // done this chunk
          break;
        }
      }

      Tensor* output_handle, *num_recs_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({2}), &output_handle));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_recs", TensorShape({}), &num_recs_t));
      auto output_vector = output_handle->vec<string>();
      auto num_recs_out = num_recs_t->scalar<int>();
      if (current_chunk_size <= 0) {
        OP_REQUIRES_OK(ctx, errors::OutOfRange("Merge op has merged all input files."));
      } 
      
      output_vector(0) = bl_ctr->container();
      output_vector(1) = bl_ctr->name();
      num_recs_out() = current_chunk_size;


    }

  private:
    ReferencePool<BufferList> *buflist_pool_ = nullptr;
    int chunk_size_;
    string path_;
    vector<string> intermediate_files_;
    vector<int> num_records_;
    vector<unique_ptr<ReadOnlyMemoryRegion>> mapped_files_;
    vector<ColumnCursor> columns_;
    priority_queue<GenomeScore, vector<GenomeScore>, ScoreComparator> score_heap_;

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      return Status::OK();
    }

  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeOp);
} // namespace tensorflow {
