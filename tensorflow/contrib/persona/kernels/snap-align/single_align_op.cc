#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>
#include <atomic>
#include <locale>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "SnapAlignerWrapper.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/contrib/persona/kernels/snap-align/single_executor.h"

namespace tensorflow {
using namespace std;
using namespace errors;

class SnapAlignSingleOp : public OpKernel {
  public:
    explicit SnapAlignSingleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));
      resource_container_shape_ = TensorShape({max_secondary_+1, 2});
    }

    ~SnapAlignSingleOp() override {
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      core::ScopedUnref executor_resource_unref(executor_resource_);
    }

  void Compute(OpKernelContext* ctx) override {
    if (!executor_resource_) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
    }

    ResourceContainer<ReadResource> *reads_container;
    OP_REQUIRES_OK(ctx, GetInput(ctx, &reads_container));
    auto reads = reads_container->get();

    vector <ResourceContainer<BufferList>*> result_buffers(max_secondary_+1);
    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx, result_buffers));
    buffer_lists_.clear();
    for (auto rc : result_buffers) {
      buffer_lists_.push_back(rc->get());
    }

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
    OP_REQUIRES_OK(ctx, ctx->allocate_output("result_buf_handle", resource_container_shape_, &out_t));
    auto out_matrix = out_t->matrix<string>();
    for (int i = 0; i < max_secondary_+1; i++) {
      out_matrix(i, 0) = result_buffers[i]->container();
      out_matrix(i, 1) = result_buffers[i]->name();
    }

    n.WaitForNotification();
  }

private:
  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  BasicContainer<SnapSingleExecutor> *executor_resource_ = nullptr;
  SnapSingleExecutor* executor_;
  int subchunk_size_, max_secondary_;
  vector <BufferList*> buffer_lists_; // just used as a cache to proxy the ResourceContainer<BufferList> instances to split()
  mutex mu_;
  TensorShape resource_container_shape_;

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "executor_handle", &executor_resource_));
    executor_ = executor_resource_->get();


    return Status::OK();
  }

  Status GetInput(OpKernelContext* ctx, ResourceContainer<ReadResource> **reads_container) {
    const Tensor *input;
    TF_RETURN_IF_ERROR(ctx->input("read", &input));
    auto data = input->vec<string>();
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
    core::ScopedUnref a(*reads_container);

    return Status::OK();
  }

  Status GetResultBufferLists(OpKernelContext* ctx, vector<ResourceContainer<BufferList>*> &result_buffers) {
    ResourceContainer<BufferList> *ctr;
    result_buffers.clear();
    for (int i = 0; i < max_secondary_+1; i++) {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(&ctr));
      ctr->get()->reset();
      result_buffers.push_back(ctr);
    }

    return Status::OK();
  }


  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignSingleOp);
};

  REGISTER_KERNEL_BUILDER(Name("SnapAlignSingle").Device(DEVICE_CPU), SnapAlignSingleOp);
}  // namespace tensorflow
