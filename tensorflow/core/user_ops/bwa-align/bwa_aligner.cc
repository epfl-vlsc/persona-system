#include <sys/time.h>
#include <sys/resource.h>
#include <array>
#include <vector>
#include <tuple>
#include <thread>
#include <memory>
#include <chrono>
#include <atomic>
#include <locale>
#include <pthread.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/agd-format/buffer_list.h"
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"
#include "bwa_wrapper.h"
#include "work_queue.h"
#include "tensorflow/core/user_ops/agd-format/read_resource.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
using namespace std;
using namespace errors;

  namespace {
    void resource_releaser(ResourceContainer<ReadResource> *rr) {
      ResourceReleaser<ReadResource> a(*rr);
      {
        ReadResourceReleaser r(*rr->get());
      }
    }

    void no_resource_releaser(ResourceContainer<ReadResource> *rr) {
      // nothing to do
    }
  }

class AGDBwaAlignerOp : public OpKernel {
  public:
    explicit AGDBwaAlignerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_read_size", &max_read_size_));
      subchunk_size_ *= 2;

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new WorkQueue<shared_ptr<ResourceContainer<ReadResource>>>(capacity));
      compute_status_ = Status::OK();
    }

    ~AGDBwaAlignerOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~AGDBwaAlignerOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_active_threads_.load() > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
      LOG(INFO) << "request queue push wait: " << request_queue_->num_push_waits();
      LOG(INFO) << "request queue pop wait: " << request_queue_->num_pop_waits();
      LOG(INFO) << "request queue peek wait: " << request_queue_->num_peek_waits();
      auto avg_inter_time = total_usec / total_invoke_intervals;
      LOG(INFO) << "average inter kernel time: " << avg_inter_time;
      //LOG(INFO) << "done queue push wait: " << done_queue_->num_push_waits();
      //LOG(INFO) << "done queue pop wait: " << done_queue_->num_pop_waits();
      VLOG(DEBUG) << "AGD Align Destructor(" << this << ") finished\n";
    }

  void Compute(OpKernelContext* ctx) override {
    if (first) {
      first = false;
    } else {
      t_now = std::chrono::high_resolution_clock::now();
      auto interval_time = std::chrono::duration_cast<std::chrono::microseconds>(t_now - t_last);
      total_usec += interval_time.count();
      total_invoke_intervals++;
    }

    if (index_resource_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
      init_workers(ctx);
    }

    if (!compute_status_.ok()) {
      ctx->SetStatus(compute_status_);
      return;
    }

    OP_REQUIRES(ctx, run_, Internal("One of the aligner threads triggered a shutdown of the aligners. Please inspect!"));

    ResourceContainer<ReadResource> *reads_container;
    OP_REQUIRES_OK(ctx, GetInput(ctx, "read", &reads_container));

    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    ResourceContainer<BufferList> *bufferlist_resource_container;
    OP_REQUIRES_OK(ctx, GetResultBufferList(ctx, &bufferlist_resource_container));

    auto alignment_result_buffer_list = bufferlist_resource_container->get();

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, alignment_result_buffer_list));

    OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, resource_releaser)), Internal("Unable to push item onto work queue. Is it already closed?"));
    t_last = std::chrono::high_resolution_clock::now();
  }

private:
  uint64 total_usec = 0;
  uint64 total_invoke_intervals = 0;
  bool first = true;
  std::chrono::high_resolution_clock::time_point t_now;
  std::chrono::high_resolution_clock::time_point t_last;

  struct time_log {
    std::chrono::high_resolution_clock::time_point end_subchunk;
    std::chrono::high_resolution_clock::time_point start_subchunk;
    std::chrono::high_resolution_clock::time_point ready;
    std::chrono::high_resolution_clock::time_point getnext;
    std::chrono::high_resolution_clock::time_point dropifequal;
    std::chrono::high_resolution_clock::time_point peek;

    void print() {

      auto subchunktime = std::chrono::duration_cast<std::chrono::microseconds>(start_subchunk - end_subchunk);
      auto readytime = std::chrono::duration_cast<std::chrono::microseconds>(ready - end_subchunk);
      auto getnexttime = std::chrono::duration_cast<std::chrono::microseconds>(getnext - ready);
      auto dropifequaltime = std::chrono::duration_cast<std::chrono::microseconds>(dropifequal - getnext);
      auto peektime = std::chrono::duration_cast<std::chrono::microseconds>(peek - dropifequal);
      LOG(INFO) << "subchunk time: " << subchunktime.count()
        << " ready time: " << readytime.count()
        << " getnext time: " << getnexttime.count()
        << " dropifequal time: " << dropifequaltime.count()
        << " peek time: " << peektime.count();
    }
  };

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "index_handle", &index_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));

    bwa_options_ = options_resource_->get();
    bwa_index_ = index_resource_->get();

    /*if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
      num_secondary_alignments_ = 0;
      } else {
      num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine,
      options_->seedCoverage, MAX_READ_LENGTH, options_->maxHits, index_resource_->get_index()->getSeedLength());
      }*/

    return Status::OK();
  }

  Status GetInput(OpKernelContext *ctx, const string &input_name, ResourceContainer<ReadResource> **reads_container)
  {
    const Tensor *read_input;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
    return Status::OK();
  }

  Status GetResultBufferList(OpKernelContext* ctx, ResourceContainer<BufferList> **ctr)
  {
    TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
    (*ctr)->get()->reset();
    TF_RETURN_IF_ERROR((*ctr)->allocate_output("result_buf_handle", ctx));
    return Status::OK();
  }

  inline void init_workers(OpKernelContext* ctx) {
    auto aligner_func = [this] () {
      std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = thread_id_++;
      }

      /*cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
      int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        LOG(INFO) << "Error calling pthread_setaffinity_np: " << rc << ", to core:0 "
          << " for thread id: " << my_id;
      } else
        LOG(INFO) << "set affinity to core 0";*/
      int capacity = request_queue_->capacity();
      //LOG(INFO) << "aligner thread spinning up";

      bwa_wrapper::BWAAligner aligner(bwa_options_, bwa_index_, max_read_size_);


      AlignmentResultBuilder result_builder;

      BufferPair* result_buf = nullptr;
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
      //std::chrono::high_resolution_clock::time_point end_subchunk = std::chrono::high_resolution_clock::now();
      //std::chrono::high_resolution_clock::time_point start_subchunk = std::chrono::high_resolution_clock::now();

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<ReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        //timeLog.peek = std::chrono::high_resolution_clock::now();

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
        if (io_chunk_status.ok()) {

          result_builder.set_buffer_pair(result_buf);

          Status s = aligner.AlignSubchunk(subchunk_resource, result_builder);

          if (!s.ok()){
            compute_status_ = s;
            return;
          }

          result_buf->set_ready();

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
        }

        request_queue_->drop_if_equal(reads_container);

        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }
      }

      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_--;
    };

    auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    for (int i = 0; i < num_threads_; i++)
      worker_threadpool->Schedule(aligner_func);
    num_active_threads_ = num_threads_;
  }

  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  BasicContainer<bwaidx_t> *index_resource_ = nullptr;
  BasicContainer<mem_opt_t>* options_resource_ = nullptr;
  bwaidx_t* bwa_index_ = nullptr;
  mem_opt_t *bwa_options_ = nullptr;
  int subchunk_size_;
  volatile bool run_ = true;
  uint64_t id_ = 0;

  atomic<uint32_t> num_active_threads_;
  mutex mu_;
  int thread_id_ = 0;
  int max_read_size_;

  int num_threads_;

  unique_ptr<WorkQueue<shared_ptr<ResourceContainer<ReadResource>>>> request_queue_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(AGDBwaAlignerOp);
};

  REGISTER_OP("AGDBwaAligner")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_read_size: int = 400")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.

Subchunk Size is the size in paired records. The actual chunk size will be 2x because of the pairing.
)doc");

  REGISTER_KERNEL_BUILDER(Name("AGDBwaAligner").Device(DEVICE_CPU), AGDBwaAlignerOp);

}  // namespace tensorflow
