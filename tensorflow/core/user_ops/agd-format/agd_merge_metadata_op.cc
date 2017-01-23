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
    const string op_name("AGDMergeMetadata");

    void resource_releaser(ResourceContainer<Data> *data) {
      core::ScopedUnref a(data);
      data->release();
    }

    class ColumnCursor {
    public:
      ColumnCursor(AGDRecordReader &&metadata, vector<AGDRecordReader> &&other_columns) :
        metadata_(move(metadata)), other_columns_(move(other_columns)) {}

      Status set_current_string() {
        const char* data;
        size_t data_sz;
        TF_RETURN_IF_ERROR(metadata_.PeekNextRecord(&data, &data_sz));
        current_meta_ = data;
        current_size_ = data_sz;
        return Status::OK();
      }

      Status append_to_buffer_list(BufferList *bl) {
        const char* data;

        // first, dump the alignment result in the first column
        auto &bp_metadata = (*bl)[0];
        TF_RETURN_IF_ERROR(copy_record(bp_metadata, metadata_));

        size_t bl_idx = 1;
        for (auto &r : other_columns_) {
          auto &bp = (*bl)[bl_idx++];
          TF_RETURN_IF_ERROR(copy_record(bp, r));
        }

        return Status::OK();
      }

      inline const char* get_string(size_t &size) {
        size = current_size_;
        return current_meta_;
      }

    private:
      vector<AGDRecordReader> other_columns_;
      AGDRecordReader metadata_;
      //const AlignmentResult *current_result_ = nullptr;
      const char * current_meta_ = nullptr;
      size_t current_size_;
      //int64_t current_location_ = -2048;

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

    typedef tuple<const char*, size_t, ColumnCursor*> MetadataScore;
    struct ScoreComparator {
      bool operator()(const MetadataScore &a, const MetadataScore &b) {
        return strncmp(get<0>(a), get<0>(b), min(get<1>(a), get<1>(b))) > 0;
      }
    };

  }

  REGISTER_OP(op_name.c_str())
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

  class AGDMergeMetadataOp : public OpKernel {
  public:
    AGDMergeMetadataOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~AGDMergeMetadataOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      const Tensor *chunk_group_handles_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_group_handles", &chunk_group_handles_t));
      auto chunk_group_shape = chunk_group_handles_t->shape();
      auto num_super_chunks = chunk_group_shape.dim_size(0);
      auto num_columns = chunk_group_shape.dim_size(1);
      auto chunk_group_handles = chunk_group_handles_t->tensor<string, 3>();

      auto rsrc_mgr = ctx->resource_manager();

      vector<ColumnCursor> columns;
      vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>> releasers;

      // Note: we don't keep the actual ColumnCursors in here. all the move and copy ops would get expensive!
      priority_queue<MetadataScore, vector<MetadataScore>, ScoreComparator> score_heap;

      releasers.reserve(num_super_chunks * num_columns);
      columns.reserve(num_super_chunks);
      ResourceContainer<Data> *data;
      bool success = false;
      const char *meta;
      size_t size;

      decltype(num_columns) column;
      for (decltype(num_super_chunks) super_chunk = 0; super_chunk < num_super_chunks; ++super_chunk) {
        column = 0;
        // First, we look up the metadata column
        OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                             chunk_group_handles(super_chunk, column, 1), &data));
        AGDRecordReader metadata_column{ AGDRecordReader::fromUncompressed(data, &success) };
        OP_REQUIRES(ctx, success, Internal("Unable to parse Metadata fromUncompressed for Metadata Merge"));
        releasers.push_back(move(decltype(releasers)::value_type(data, resource_releaser)));

        // Then we look up the rest of the columns
        vector<AGDRecordReader> other_columns;
        other_columns.reserve(num_columns-1);
        for (column = 1; column < num_columns; ++column) {
          OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                               chunk_group_handles(super_chunk, column, 1), &data));
          AGDRecordReader other_column{ AGDRecordReader::fromUncompressed(data, &success) };
          OP_REQUIRES(ctx, success, Internal("Unable to parse other column fromUncompressed for Metadata Merge"));
          other_columns.push_back(move(other_column));
          releasers.push_back(move(decltype(releasers)::value_type(data, resource_releaser)));
        }

        ColumnCursor a(move(metadata_column), move(other_columns));
        OP_REQUIRES_OK(ctx, a.set_current_string());
        columns.push_back(move(a));
      }

      // Now that everything is initialized, add the scores to the heap
      for (auto &cc : columns) {
        meta = cc.get_string(size);
        score_heap.push(MetadataScore(meta, size, &cc));
      }

      int current_chunk_size = 0;
      ColumnCursor *cc;
      ResourceContainer<BufferList> *bl_ctr;
      OP_REQUIRES_OK(ctx, buflist_pool_->GetResource(&bl_ctr));
      auto bl = bl_ctr->get();
      bl->resize(num_columns);
      Status s;
      while (!score_heap.empty()) {
        auto &top = score_heap.top();
        cc = get<2>(top);

        cc->append_to_buffer_list(bl);

        score_heap.pop();

        s = cc->set_current_string();
        if (s.ok()) {
          // get_location will have the location advanced by the append_to_buffer_list call above
          meta = cc->get_string(size);
          score_heap.push(MetadataScore(meta, size, cc));
        } else if (!IsResourceExhausted(s)) {
          OP_REQUIRES_OK(ctx, s);
        } 

        // pre-increment because we just added 1 to the chunk size
        // we're guaranteed that chunk size is at least 1
        if (++current_chunk_size == chunk_size_) {
          OP_REQUIRES_OK(ctx, EnqueueBufferList(ctx, bl_ctr, current_chunk_size));
          OP_REQUIRES_OK(ctx, buflist_pool_->GetResource(&bl_ctr));
          bl = bl_ctr->get();
          bl->resize(num_columns);
          current_chunk_size = 0;
        }
      }

      if (current_chunk_size > 0) {
        OP_REQUIRES_OK(ctx, EnqueueBufferList(ctx, bl_ctr, current_chunk_size));
      }

      // Not sure if needed when using a queue runner?
      //queue_->Close(ctx, false, [](){});
    }

  private:
    QueueInterface *queue_ = nullptr;
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
    ReferencePool<BufferList> *buflist_pool_ = nullptr;
    TensorShape enqueue_shape_{{2}}, num_records_shape_{};
    int chunk_size_;

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      return Status::OK();
    }

    Status EnqueueBufferList(OpKernelContext *ctx, ResourceContainer<BufferList> *bl_ctr, size_t chunk_size) {
      QueueInterface::Tuple tuple; // just a vector<Tensor>
      Tensor container_out, num_recs_out;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &container_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, num_records_shape_, &num_recs_out));
      auto container_out_vec = container_out.vec<string>();
      num_recs_out.scalar<int>()() = chunk_size;
      tuple.push_back(num_recs_out);
      container_out_vec(0) = bl_ctr->container();
      container_out_vec(1) = bl_ctr->name();
      tuple.push_back(container_out); // performs a shallow copy. Destructor doesn't release resources

      TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));

      // This is the synchronous version
      /*
      Notification n;
      queue_->TryEnqueue(tuple, ctx, [&n]() { n.Notify(); });
      n.WaitForNotification();
      */
      queue_->TryEnqueue(tuple, ctx, [](){});

      return Status::OK();
    }

  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeMetadataOp);
} // namespace tensorflow {
