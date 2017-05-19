#include <vector>
#include <memory>
#include <utility>
#include <queue>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"

#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  inline bool operator>(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() > rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() > rhs.position()) return true;
      else return false;
    } else
      return false;
  }

  namespace {
    const string op_name("AGDCephMerge");

    class ColumnCursor {
    public:
      ColumnCursor(unique_ptr<AGDRemoteRecordReader> &&results, 
          vector<unique_ptr<AGDRemoteRecordReader>> &&other_columns) :
        results_(move(results)), other_columns_(move(other_columns)) {
      }

      Status Initialize() {
        Status s = results_->Initialize(); 
        if (!s.ok())
          return s;
        for (auto& col : other_columns_) {
          s = col->Initialize();
          if (!s.ok())
            return s;
        }
        return Status::OK();
      }

      Status set_current_location() {
        const char* data;
        size_t data_sz;
        TF_RETURN_IF_ERROR(results_->PeekNextRecord(&data, &data_sz));
        current_result_.ParseFromArray(data, data_sz);
        current_position_ = current_result_.position();
        return Status::OK();
      }

      Status append_to_buffer_list(BufferList *bl) {
        const char* data;

        // first, dump the alignment result in the first column
        auto &bp_results = (*bl)[0];
        TF_RETURN_IF_ERROR(copy_record(bp_results, results_.get()));

        size_t bl_idx = 1;
        for (auto &r : other_columns_) {
          auto &bp = (*bl)[bl_idx++];
          TF_RETURN_IF_ERROR(copy_record(bp, r.get()));
        }

        return Status::OK();
      }

      inline Position get_location() {
        return current_position_;
      }

      // this is supposed to be called from another thread because
      // the main will block on RemoteRecordReader::GetNextRecord
      // if there is no data
      void PrefetchData() {
        Status s = results_->PrefetchRecords();
        if (!IsResourceExhausted(s))
          if (!s.ok())
            LOG(INFO) << "Prefetching ceph data failed: " << s;
        for (auto& reader : other_columns_) {
          s = reader->PrefetchRecords();
          if (!IsResourceExhausted(s))
            if (!s.ok())
              LOG(INFO) << "Prefetching ceph data failed: " << s;
        }
      }

    private:
      vector<unique_ptr<AGDRemoteRecordReader>> other_columns_;
      unique_ptr<AGDRemoteRecordReader> results_;
      Alignment current_result_;
      Position current_position_;

      static inline
      Status
      copy_record(BufferPair& bp, AGDRemoteRecordReader *r) {
        const char *record_data;
        size_t record_size;
        auto &index = bp.index();
        auto &data = bp.data();

        TF_RETURN_IF_ERROR(r->GetNextRecord(&record_data, &record_size));
        auto char_sz = static_cast<char>(record_size);
        TF_RETURN_IF_ERROR(index.AppendBuffer(&char_sz, sizeof(char_sz)));
        TF_RETURN_IF_ERROR(data.AppendBuffer(record_data, record_size));

        return Status::OK();
      }
    };

    typedef pair<Position, ColumnCursor*> GenomeScore;
    struct ScoreComparator {
      bool operator()(const GenomeScore &a, const GenomeScore &b) {
        return a.first > b.first;
      }
    };

  }


  class AGDCephMergeOp : public OpKernel {
  public:
    AGDCephMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("intermediate_files", &intermediate_files_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_records", &num_records_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("file_buf_size", &file_buf_size_));
      file_buf_size_ *= 1000000; // convert to bytes

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster_.init2(user_name_.c_str(), cluster_name_.c_str(), 0);
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster init2\nUsername: ", user_name_, "\nCluster Name: ", cluster_name_, "\nReturn code: ", ret));

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf_));
      ret = cluster_.conf_read_file(ceph_conf_.c_str());
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf_, "' returned ", ret, " when attempting to open"));

      /* Connect to the cluster */
      ret = cluster_.connect();
      OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));

      /* Set up IO context */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name_));
      
      ret = cluster_.ioctx_create(pool_name_.c_str(), io_ctx_);
      OP_REQUIRES(ctx, ret == 0, Internal("Unable to create Ceph io ctx with pool name ", pool_name_, 
            " returned ", ret));

      uint64_t size =  file_buf_size_*4*intermediate_files_.size();
      LOG(INFO) << "allocating " << size << " bytes of memory for merge";
      big_buf_.reset(new char[size]);
      char* the_ptr = big_buf_.get();
      LOG(INFO) << "done";

      for (size_t i = 0; i < intermediate_files_.size(); i++) {
        string bases_file = intermediate_files_[i] + ".base";
        string meta_file = intermediate_files_[i] + ".metadata";
        string qual_file = intermediate_files_[i] + ".qual";
        string results_file = intermediate_files_[i] + ".results";

        // RemoteRecordReader owns nothing
        auto results_column = unique_ptr<AGDRemoteRecordReader>(new AGDRemoteRecordReader(results_file, num_records_[i], the_ptr, 
            file_buf_size_, &io_ctx_));

        the_ptr += file_buf_size_;
        vector<unique_ptr<AGDRemoteRecordReader>> other_columns;
        other_columns.push_back(move(unique_ptr<AGDRemoteRecordReader>(new AGDRemoteRecordReader(bases_file, num_records_[i], 
              the_ptr, file_buf_size_, &io_ctx_))));
        the_ptr += file_buf_size_;
        other_columns.push_back(move(unique_ptr<AGDRemoteRecordReader>(new AGDRemoteRecordReader(meta_file, num_records_[i], 
              the_ptr, file_buf_size_, &io_ctx_))));
        the_ptr += file_buf_size_;
        other_columns.push_back(move(unique_ptr<AGDRemoteRecordReader>(new AGDRemoteRecordReader(qual_file, num_records_[i], 
              the_ptr, file_buf_size_, &io_ctx_))));
        the_ptr += file_buf_size_;

        ColumnCursor a(move(results_column), move(other_columns));
        //OP_REQUIRES_OK(ctx, a.set_current_location());
        columns_.push_back(move(a));
      }
      
      LOG(INFO) << "done merge setup";


    }

    ~AGDCephMergeOp() {
      run_ = false;
      while (thread_active_);

    }

    void Compute(OpKernelContext* ctx) override {
      if (!buflist_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
        run_ = true;
        auto filler_func = [this]() {
          while (run_) {
            for (auto& cc : columns_) 
              cc.PrefetchData();
          }
          thread_active_ = false;
        };
        thread_active_ = true;
        auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
        worker_threadpool->Schedule(filler_func);
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

        //LOG(INFO) << "processing location: " << top.first;
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
    vector<string> intermediate_files_;
    vector<int> num_records_;
    vector<ColumnCursor> columns_;
    priority_queue<GenomeScore, vector<GenomeScore>, ScoreComparator> score_heap_;
    librados::IoCtx io_ctx_;
    string cluster_name_;
    string user_name_;
    string pool_name_;
    string ceph_conf_;
    librados::Rados cluster_;
    int64 file_buf_size_;
    unique_ptr<char[]> big_buf_;
    volatile bool thread_active_;
    volatile bool run_;

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      for (auto& col : columns_) {
        TF_RETURN_IF_ERROR(col.Initialize());
        TF_RETURN_IF_ERROR(col.set_current_location());
        score_heap_.push(GenomeScore(col.get_location(), &col));
      }
      return Status::OK();
    }

  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDCephMergeOp);
} // namespace tensorflow {
