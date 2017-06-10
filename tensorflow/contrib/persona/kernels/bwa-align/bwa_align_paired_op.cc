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
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "bwa_wrapper.h"
#include "bwa_reads.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_paired_executor.h"

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

  class BWAAlignPairedOp : public OpKernel {
    public:
      explicit BWAAlignPairedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_read_size", &max_read_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));
        subchunk_size_ *= 2;

        int capacity;
      }

      ~BWAAlignPairedOp() override {
        core::ScopedUnref buflist_pool_unref(buflist_pool_);
      }

      void Compute(OpKernelContext* ctx) override {
        if (executor_resource_ == nullptr) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }


        //ResourceContainer<BufferList> *bufferlist_resource_container;
        OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx));

        ResourceContainer<BWAReadResource> *reads_container;
        OP_REQUIRES_OK(ctx, GetInput(ctx, "read", &reads_container));

        // dont want to delete yet
        core::ScopedUnref a(reads_container);
        auto reads = reads_container->get();

        OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));

        Notification n;
        OP_REQUIRES_OK(ctx, executor_->EnqueueChunk(shared_ptr<ResourceContainer<BWAReadResource>>(
                reads_container, [this, ctx, &n](ResourceContainer<BWAReadResource> *rr) {
                  ResourceReleaser<BWAReadResource> a(*rr);
                  {
                    ReadResourceReleaser r(*rr->get());
                  }
                  n.Notify();
                }
        )));


        n.WaitForNotification();
        //t_last = std::chrono::high_resolution_clock::now();
      }

    private:
      uint64 total_usec = 0;
      uint64 total_invoke_intervals = 0;
      std::chrono::high_resolution_clock::time_point t_now;
      std::chrono::high_resolution_clock::time_point t_last;

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "executor_handle", &executor_resource_));
        executor_ = executor_resource_->get();

        return Status::OK();
      }

      Status GetInput(OpKernelContext *ctx, const string &input_name, ResourceContainer<BWAReadResource> **reads_container)
      {
        const Tensor *read_input;
        TF_RETURN_IF_ERROR(ctx->input(input_name, &read_input));
        auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
        TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));

        return Status::OK();
      }

      Status GetResultBufferLists(OpKernelContext* ctx)
      {
        ResourceContainer<BufferList> *ctr;
        Tensor* out_t;
        buffer_lists_.clear();
        buffer_lists_.reserve(max_secondary_+1);
        TF_RETURN_IF_ERROR(ctx->allocate_output("result_buf_handle", TensorShape({max_secondary_+1, 2}), &out_t));
        auto out_matrix = out_t->matrix<string>();
        for (int i = 0; i < max_secondary_+1; i++) {
          TF_RETURN_IF_ERROR(buflist_pool_->GetResource(&ctr));
          //core::ScopedUnref a(reads_container);
          ctr->get()->reset();
          buffer_lists_.push_back(ctr->get());
          out_matrix(i, 0) = ctr->container();
          out_matrix(i, 1) = ctr->name();
        }

        return Status::OK();
      }

      BasicContainer<BWAPairedExecutor> *executor_resource_ = nullptr;
      BWAPairedExecutor* executor_;

      ReferencePool<BufferList> *buflist_pool_ = nullptr;
      int subchunk_size_;
      volatile bool run_ = true;
      uint64_t id_ = 0;

      int max_read_size_;
      int max_secondary_;

      vector<BufferList*> buffer_lists_;


      TF_DISALLOW_COPY_AND_ASSIGN(BWAAlignPairedOp);
  };

  REGISTER_KERNEL_BUILDER(Name("BWAAlignPaired").Device(DEVICE_CPU), BWAAlignPairedOp);

}  // namespace tensorflow
