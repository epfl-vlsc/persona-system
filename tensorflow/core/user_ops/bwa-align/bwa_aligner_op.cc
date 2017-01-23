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
#include "bwa_reads.h"
#include "work_queue.h"
#include "tensorflow/core/user_ops/agd-format/read_resource.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
using namespace std;
using namespace errors;

  namespace {
    void resource_releaser(ResourceContainer<BWAReadResource> *rr) {
      ResourceReleaser<BWAReadResource> a(*rr);
      {
        ReadResourceReleaser r(*rr->get());
      }
    }

    void no_resource_releaser(ResourceContainer<BWAReadResource> *rr) {
      // nothing to do
    }
  }

class BWAAlignerOp : public OpKernel {
  public:
    explicit BWAAlignerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_read_size", &max_read_size_));
      subchunk_size_ *= 2;

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new WorkQueue<shared_ptr<ResourceContainer<BWAReadResource>>>(capacity));
      compute_status_ = Status::OK();
    }

    ~BWAAlignerOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~BWAAlignerOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      //core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_active_threads_.load() > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
      LOG(INFO) << "request queue push wait: " << request_queue_->num_push_waits();
      LOG(INFO) << "request queue pop wait: " << request_queue_->num_pop_waits();
      LOG(INFO) << "request queue peek wait: " << request_queue_->num_peek_waits();
      uint64_t avg_inter_time = 0;
      if (total_usec > 0)
        avg_inter_time = total_usec / total_invoke_intervals;

      LOG(INFO) << "average inter kernel time: " << avg_inter_time ? to_string(avg_inter_time) : "n/a";;
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

    ResourceContainer<BWAReadResource> *reads_container;
    OP_REQUIRES_OK(ctx, GetInput(ctx, "read", &reads_container));

    LOG(INFO) << "aligner got reads";
    // dont want to delete yet
    //core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, nullptr)); // no bufferlist

    OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<BWAReadResource>>(reads_container, no_resource_releaser)), 
        Internal("Unable to push item onto work queue. Is it already closed?"));
    t_last = std::chrono::high_resolution_clock::now();
  }

private:
  uint64 total_usec = 0;
  uint64 total_invoke_intervals = 0;
  bool first = true;
  std::chrono::high_resolution_clock::time_point t_now;
  std::chrono::high_resolution_clock::time_point t_last;

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "index_handle", &index_resource_));

    bwa_options_ = options_resource_->get();
    bwa_index_ = index_resource_->get();

    return Status::OK();
  }

  Status GetInput(OpKernelContext *ctx, const string &input_name, ResourceContainer<BWAReadResource> **reads_container)
  {
    const Tensor *read_input;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
   
    // input is output as well
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

      bwa_wrapper::BWAAligner aligner(bwa_options_, bwa_index_, max_read_size_);

      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
      //std::chrono::high_resolution_clock::time_point end_subchunk = std::chrono::high_resolution_clock::now();
      //std::chrono::high_resolution_clock::time_point start_subchunk = std::chrono::high_resolution_clock::now();

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<BWAReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        //timeLog.peek = std::chrono::high_resolution_clock::now();

        auto *reads = reads_container->get();

        size_t interval;
        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &interval);
        vector<mem_alnreg_v>& regs = reads->get_regs();
        while (io_chunk_status.ok()) {
          LOG(INFO) << "aligner thread " << my_id << " got interval " << interval;


          Status s = aligner.AlignSubchunk(subchunk_resource, interval, regs);

          if (!s.ok()){
            compute_status_ = s;
            return;
          }

          reads->decrement_outstanding();

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &interval);
        }
        
        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }

        request_queue_->drop_if_equal(reads_container);

      }

      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_--;
    };

    auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    for (int i = 0; i < num_threads_; i++)
      worker_threadpool->Schedule(aligner_func);
    num_active_threads_ = num_threads_;
  }

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

  unique_ptr<WorkQueue<shared_ptr<ResourceContainer<BWAReadResource>>>> request_queue_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(BWAAlignerOp);
};

  REGISTER_OP("BWAAligner")
  .Attr("num_threads: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 3")
  .Attr("max_read_size: int = 400")
  .Input("index_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("read: string")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
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

  REGISTER_KERNEL_BUILDER(Name("BWAAligner").Device(DEVICE_CPU), BWAAlignerOp);

}  // namespace tensorflow
