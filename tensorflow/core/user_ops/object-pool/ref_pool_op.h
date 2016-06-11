#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "ref_pool.h"
#include "resource_container.h"
#include <string>
#include <utility>
#include <memory>

namespace tensorflow {

  // let the consumer write their own doc
#define REGISTER_REFERENCE_POOL(_NAME) \
  REGISTER_OP(_NAME) \
    .Attr("size: int") \
    .Attr("container: string = ''") \
    .Attr("shared_name: string = ''") \
    .Output("pool_handle: Ref(string)") \
    .SetIsStateful()

template <typename T>
class ReferencePoolOp : public OpKernel {

public:

  ReferencePoolOp(OpKernelConstruction* context) : OpKernel(context) {
    using namespace errors;

    OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
    OP_REQUIRES(context, size_ > 0, InvalidArgument("ReferencePoolOp: size must be >0: ", size_));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                         &pool_handle_, nullptr));
  }

  ~ReferencePoolOp() override {
    mutex_lock l(mu_);
    if (pool_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->Delete<ReferencePool<T>>(cinfo_.container(), cinfo_.name()));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    {
      mutex_lock l(mu_);
      if (!pool_handle_set_) {
        OP_REQUIRES_OK(ctx, CreatePool(ctx));
      }
    }
    OP_REQUIRES_OK(ctx, ctx->set_output_ref("pool_handle", &mu_, pool_handle_.AccessTensor(ctx)));
  }

protected:
  Status CreatePool(OpKernelContext *ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    auto rmgr = cinfo_.resource_manager();

    std::unique_ptr<ReferencePool<T>> ref_pool(new ReferencePool<T>());

    string s;
    std::unique_ptr<ResourceContainer<T>> a;
    std::unique_ptr<T> obj;
    for (int i = 0; i < size_; i++) {
      // make the name
      s = cinfo_.name();
      s.append("-");
      s.append(std::to_string(i));
      obj = CreateObject();
      a.reset(new ResourceContainer<T>(std::move(obj), cinfo_.container(), s, ref_pool.get()));
      TF_RETURN_IF_ERROR(rmgr->Create<ResourceContainer<T>>(cinfo_.container(), s, a.get()));
      ref_pool->AddResource(std::move(a));
    }

    auto h = pool_handle_.AccessTensor(ctx)->vec<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();

    // put ref_pool into the shared resource
    TF_RETURN_IF_ERROR(rmgr->Create<ReferencePool<T>>(cinfo_.container(), cinfo_.name(), ref_pool.release()));
    pool_handle_set_ = true;
  }

  virtual std::unique_ptr<T> CreateObject() = 0;

private:

  mutable mutex mu_;
  int size_;
  ContainerInfo cinfo_;
  PersistentTensor pool_handle_ GUARDED_BY(mu_);
  bool pool_handle_set_ GUARDED_BY(mu_);
};

} // namespace tensorflow {
