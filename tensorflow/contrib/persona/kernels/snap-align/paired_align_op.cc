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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "SnapAlignerWrapper.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "paired_executor.h"

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
  }

class SnapAlignPairedOp : public OpKernel {
  public:
    explicit SnapAlignPairedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));
      subchunk_size_ *= 2;
      size_t num_columns = max_secondary_ + 1;
      buffer_lists_.resize(num_columns);

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>(capacity));
      compute_status_ = Status::OK();
    }

    ~SnapAlignPairedOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~SnapAlignPairedOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_active_threads_.load() > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
    }

  void Compute(OpKernelContext* ctx) override {
    if (!index_resource_) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
    }

    ResourceContainer<ReadResource> *reads_container;
    OP_REQUIRES_OK(ctx, GetInput(ctx, &reads_container));

    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();
    auto num_records = reads->num_records();
    OP_REQUIRES(ctx, (num_records % subchunk_size_) % 2 == 0,
                Internal("Uneven number of records for chunk size ", num_records, " and subchunk size ", subchunk_size_));

    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx));

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));

    Notification n;
    OP_REQUIRES_OK(ctx, executor_->EnqueueChunk(shared_ptr<ResourceContainer<ReadResource>>(
            reads_container, [this, ctx, &n](ResourceContainer<ReadResource> *rr) {
              ResourceReleaser<ReadResource> a(*rr);
              {
                ReadResourceReleaser r(*rr->get());
                n.Notify();
              }
            }
    )));

    Tensor* out_t;

    n.WaitForNotification();
  }

private:

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "executor_handle", &executor_resource_));
    executor_ = executor_resource_->get();
    return Status::OK();
  }

  Status GetInput(OpKernelContext *ctx, ResourceContainer<ReadResource> **reads_container)
  {
    const Tensor *read_input;
    TF_RETURN_IF_ERROR(ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
    return Status::OK();
  }

  Status GetResultBufferLists(OpKernelContext* ctx)
  {
    ResourceContainer<BufferList> **ctr;
    Tensor* out_t;
    TF_RETURN_IF_ERROR(ctx->allocate_output("result_buf_handle", resource_container_shape_, &out_t));
    auto out_matrix = out_t->matrix<string>();
    for (int i = 0; i < max_secondary_+1; i++) {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      buffer_lists_[i] = (*ctr)->get();
      out_matrix(i, 0) = (*ctr)->container();
      out_matrix(i, 1) = (*ctr)->name();
    }

    return Status::OK();
  }

  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  BasicContainer<GenomeIndex> *index_resource_ = nullptr;
  BasicContainer<PairedAlignerOptions>* options_resource_ = nullptr;
  BasicContainer<SnapPairedExecutor> *executor_resource_ = nullptr;
  SnapPairedExecutor *executor_;
  PairedAlignerOptions *options_ = nullptr;
  int subchunk_size_, max_secondary_;
  volatile bool run_ = true;
  uint64_t id_ = 0;
  vector<BufferList*> buffer_lists_;
  TensorShape resource_container_shape_;

  atomic<uint32_t> num_active_threads_;
  mutex mu_;

  int num_threads_;

  unique_ptr<ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>> request_queue_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignPairedOp);
};

  REGISTER_KERNEL_BUILDER(Name("SnapAlignPaired").Device(DEVICE_CPU), SnapAlignPairedOp);

}  // namespace tensorflow
