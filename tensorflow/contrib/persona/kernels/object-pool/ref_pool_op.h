#pragma once

#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/resource_mgr.h"
#include "ref_pool.h"
#include "resource_container.h"
#include <string>
#include <utility>
#include <memory>
#include <type_traits>

namespace tensorflow {

  // T is the type of the actual container being made
  // U is the container it puts the resources in (by type), so further ops can do a generic lookup
template <typename T, typename U>
class ReferencePoolOp : public OpKernel {

public:

 ReferencePoolOp(OpKernelConstruction* context) : OpKernel(context), pool_handle_set_(false) {
    static_assert(std::is_base_of<U,T>::value, "not able to construct reference pool of non-base type");
    using namespace errors;

    OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
    OP_REQUIRES(context, size_ >= 0, InvalidArgument("ReferencePoolOp: size must be >= 0: ", size_));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                         &pool_handle_, nullptr));
    OP_REQUIRES_OK(context, context->GetAttr("bound", &bound_));
  }

  ~ReferencePoolOp() override {
    mutex_lock l(mu_);
    if (pool_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->template Delete<ReferencePool<T>>(cinfo_.container(), cinfo_.name()));
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

    std::unique_ptr<ReferencePool<T>> ref_pool;
    if (bound_) {
      ref_pool.reset(new ReferencePool<T>());
    } else {
      auto func = [this](ReferencePool<T>* t) -> Status {
        return AddObjectToPool(t);
      };
      ref_pool.reset(new ReferencePool<T>(func));
    }

    auto rp = ref_pool.get();
    while (idx_ < size_) {
      TF_RETURN_IF_ERROR(AddObjectToPool(rp));
    }

    // put ref_pool into the shared resource
    TF_RETURN_IF_ERROR(rmgr->template Create<ReferencePool<T>>(cinfo_.container(), cinfo_.name(), ref_pool.release()));
    auto h = pool_handle_.AccessTensor(ctx)->template vec<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    pool_handle_set_ = true;
    return Status::OK();
  }

  inline Status AddObjectToPool(ReferencePool<T> *ref_pool)
  {
    // make the name
    string s(cinfo_.name());
    s.append("-");
    s.append(std::to_string(idx_++));
    auto obj = CreateObject();
    std::unique_ptr<ResourceContainer<T>> a(new ResourceContainer<T>(std::move(obj), cinfo_.container(), s, ref_pool));
    // This cast is correct because of the is_base_of check above,
    // and the fact that resource container is just a smart pointer
    TF_RETURN_IF_ERROR(cinfo_.resource_manager()->template Create<ResourceContainer<U>>(cinfo_.container(), s, reinterpret_cast<ResourceContainer<U>*>(a.get())));
    ref_pool->AddResource(std::move(a));
    return Status::OK();
  }

  virtual std::unique_ptr<T> CreateObject() = 0;

private:

  mutable mutex mu_;
  int size_, idx_ = 0;
  ContainerInfo cinfo_;
  bool bound_;
  PersistentTensor pool_handle_ GUARDED_BY(mu_);
  bool pool_handle_set_ GUARDED_BY(mu_);
};

} // namespace tensorflow {
