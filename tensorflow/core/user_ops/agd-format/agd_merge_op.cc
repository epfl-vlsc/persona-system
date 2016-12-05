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

      Status advance_record() {
        const char* data;
        TF_RETURN_IF_ERROR(results_.PeekNextRecord(&data, &current_result_size_));
        current_result_ = reinterpret_cast<decltype(current_result_)>(data);
        current_location_ = current_result_->location_;
        return Status::OK();
      }

      inline int64_t get_location() {
        return current_location_;
      }

    private:
      vector<AGDRecordReader> other_columns_;
      AGDRecordReader results_;
      const AlignmentResult *current_result_ = nullptr;
      size_t current_result_size_;
      int64_t current_location_ = -2048;

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
  .Input("buffer_pool: Ref(string)")
  .Input("num_records: int32")
  .Input("chunk_group_handles: string") // a record of NUM_SUPER_CHUNKS x NUM_COLUMNS x 2 (2 for reference)
  .Input("output_buffer_queue_handle: Ref(string)")
  .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
num_records: vector of number of records
*_handles: matrix of processed handles
output_buffer_queue_handle: a handle to a queue, into which are enqueued BufferList instance handles.
)doc");

  class AGDMergeOp : public OpKernel {
  public:
    AGDMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~AGDMergeOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      const Tensor *chunk_group_handles_t, *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_group_handles", &chunk_group_handles_t));
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto chunk_group_shape = chunk_group_handles_t->shape();
      auto num_super_chunks = chunk_group_shape.dim_size(0);
      decltype(num_super_chunks) super_chunk;
      auto num_columns = chunk_group_shape.dim_size(1);
      decltype(num_columns) column;
      auto chunk_group_handles = chunk_group_handles_t->tensor<string, 3>();
      auto num_records = num_records_t->vec<int32>();

      auto rsrc_mgr = ctx->resource_manager();

      vector<ColumnCursor> columns;
      vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>> releasers;
      priority_queue<GenomeScore, vector<GenomeScore>, ScoreComparator> score_heap;

      releasers.reserve(num_super_chunks * num_columns);
      columns.reserve(num_super_chunks);
      ResourceContainer<Data> *data;

      for (super_chunk = 0; super_chunk < num_super_chunks; ++super_chunk) {
        auto super_chunk_record_count = num_records(super_chunk);
        column = 0;
        // First, we look up the results column
        OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                             chunk_group_handles(super_chunk, column, 1), &data));
        AGDRecordReader results_column(data, super_chunk_record_count);
        releasers.push_back(move(decltype(releasers)::value_type(data, resource_releaser)));

        // Then we look up the rest of the columns
        vector<AGDRecordReader> other_columns;
        other_columns.reserve(num_columns-1);
        for (column = 1; column < num_columns; ++column) {
          OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                               chunk_group_handles(super_chunk, column, 1), &data));
          other_columns.push_back(AGDRecordReader(data, super_chunk_record_count));
          releasers.push_back(move(decltype(releasers)::value_type(data, resource_releaser)));
        }
        ColumnCursor a(move(results_column), move(other_columns));
        OP_REQUIRES_OK(ctx, a.advance_record());
        columns.push_back(move(a));
      }

      for (auto &cc : columns) {
        score_heap.push(GenomeScore(cc.get_location(), &cc));
      }

      ColumnCursor *cc;
      while (!score_heap.empty()) {
        auto &top = score_heap.top();
        cc = top.second;

        // TODO actual stuff!

        // TODO only do this if non-empty
        score_heap.push(GenomeScore(cc->get_location(), cc));
        score_heap.pop();
      }
    }

  private:
    QueueInterface *queue_ = nullptr;
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
    int chunk_size_;

    Status Init(OpKernelContext *ctx) {
      // TODO these might not be able to use the convenience method :/
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "output_buffer_queue_handle", &queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeOp);
} // namespace tensorflow {
