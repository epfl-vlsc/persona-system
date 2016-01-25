/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/data_flow_ops.cc.

#define EIGEN_USE_THREADS

#include <limits.h>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"
#include "tensorflow/core/kernels/concat_op.h"
#include "tensorflow/core/kernels/split_op.h"
#include "tensorflow/core/kernels/tensor_array.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

Status GetHandle(const string& input_name, OpKernelContext* ctx,
                 string* container, string* ta_handle) {
  {
    Tensor tensor;
    // Assuming that input_name is at position 0 for puposes of
    // has_input.
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, false));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Tensor array handle must be 2-element vector, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<string>();
    *container = h(0);
    *ta_handle = h(1);
  }
  return Status::OK();
}

Status GetTensorArray(const string& input_name, OpKernelContext* ctx,
                      TensorArray** tensor_array) {
  string container;
  string ta_handle;
  TF_RETURN_IF_ERROR(GetHandle(input_name, ctx, &container, &ta_handle));
  ResourceMgr* rm = ctx->step_resource_manager();
  if (rm == nullptr) return errors::Internal("No per-step resource manager.");
  TF_RETURN_IF_ERROR(rm->Lookup(container, ta_handle, tensor_array));
  return Status::OK();
}

Status SetupFlowControlInputs(OpKernelContext* ctx, bool set_output) {
  const Tensor* flow_control;
  TF_RETURN_IF_ERROR(ctx->input("flow_in", &flow_control));
  if (set_output) {
    TF_RETURN_IF_ERROR(ctx->set_output("flow_out", *flow_control));
  }
  return Status::OK();
}

// Virtual class for shared behavior between TensorArrayOp and
// TensorArrayGradOp.
class TensorArrayCreationOp : public OpKernel {
 public:
  explicit TensorArrayCreationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor tensor_array_output_handle;

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            tensorflow::DT_STRING, tensorflow::TensorShape({2}),
                            &tensor_array_output_handle, alloc_attr));
    // Store the handle in a container of the per-step RM.
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));

    TensorArray* output_tensor_array;
    OP_REQUIRES_OK(ctx, CreateTensorArray(ctx, rm, &tensor_array_output_handle,
                                          &output_tensor_array));

    ctx->set_output_ref(0, output_tensor_array->mu(),
                        output_tensor_array->handle());
  }

 protected:
  virtual Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                                   Tensor* tensor_array_output_handle,
                                   TensorArray** output_tensor_array) = 0;
};

// A per-run local tensor array. The tensor array uses a "per-step" resource
// manager which ensures that correct garbage collection on error or
// successful completion.
class TensorArrayOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_array_name", &tensor_array_name_));
    if (tensor_array_name_ == "") tensor_array_name_ = name();
  }

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
    const Tensor* tensor_size;
    TF_RETURN_IF_ERROR(ctx->input("size", &tensor_size));

    if (!TensorShapeUtils::IsScalar(tensor_size->shape())) {
      return errors::InvalidArgument(
          "TensorArray size must be scalar, but had shape: ",
          tensor_size->shape().DebugString());
    }
    const int32 size = tensor_size->scalar<int32>()();

    auto handle = tensor_array_output_handle->flat<string>();
    handle(0) = "_tensor_arrays";
    handle(1) = tensor_array_name_;

    TensorArray* tensor_array =
        new TensorArray(dtype_, *tensor_array_output_handle, size);

    TF_RETURN_IF_ERROR(rm->Create(handle(0), tensor_array_name_, tensor_array));

    *output_tensor_array = tensor_array;

    return Status::OK();
  }

 private:
  DataType dtype_;
  string tensor_array_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArray").Device(DEVICE_CPU), TensorArrayOp);

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArray")                \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

class TensorArrayGradOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayGradOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {}

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
    string container;
    string tensor_array_name;
    TF_RETURN_IF_ERROR(
        GetHandle("handle", ctx, &container, &tensor_array_name));

    if (container != "_tensor_arrays") {
      return errors::InvalidArgument(
          "Input container should be '_tensor_arrays',  but received '",
          container, "'");
    }

    auto output_handle = tensor_array_output_handle->flat<string>();
    output_handle(0) = "_tensor_array_grads";
    output_handle(1) = tensor_array_name;

    TensorArray* tensor_array;
    TF_RETURN_IF_ERROR(rm->Lookup(container, tensor_array_name, &tensor_array));

    auto creator = [this, tensor_array,
                    tensor_array_output_handle](TensorArray** ret) {
      *ret = new TensorArray(tensor_array->ElemType(),
                             *tensor_array_output_handle, tensor_array->Size());
      return Status::OK();
    };

    Status s = rm->LookupOrCreate<TensorArray>(
        output_handle(0), output_handle(1), output_tensor_array, creator);

    return s;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGradOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad").Device(DEVICE_CPU),
                        TensorArrayGradOp);

REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);

template <typename Device, typename T>
class TensorArrayWriteOp : public OpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("gradient_add", &gradient_add_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    const Tensor* tensor_index;
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    const int32 index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    PersistentTensor persistent_tensor(*tensor_value);
    if (gradient_add_) {
      Status s =
          tensor_array->WriteOrAdd<Device, T>(ctx, index, persistent_tensor);
      OP_REQUIRES_OK(ctx, s);
    } else {
      OP_REQUIRES_OK(ctx, tensor_array->Write(index, persistent_tensor));
    }
  }

 private:
  bool gradient_add_;
};

#define REGISTER_WRITE(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("TensorArrayWrite").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayWriteOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_WRITE);

#undef REGISTER_WRITE

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWrite")       \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("handle")      \
                              .HostMemory("index"),      \
                          TensorArrayWriteOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

class TensorArrayReadOp : public OpKernel {
 public:
  explicit TensorArrayReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    const Tensor* tensor_index;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));

    const int32 index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));
    PersistentTensor value;
    OP_REQUIRES_OK(ctx, tensor_array->Read(index, &value));
    ctx->set_output(0, *value.AccessTensor(ctx));
  }

 private:
  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayRead").Device(DEVICE_CPU),
                        TensorArrayReadOp);

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayRead")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle")          \
                              .HostMemory("index"),          \
                          TensorArrayReadOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// Concatenate the elements in a TensorArray.  All elements must be
// defined and have the same shape.
template <typename Device, typename T>
class TensorArrayPackOp : public OpKernel {
 public:
  typedef typename TTypes<T, 2>::ConstMatrix ConstMatrix;
  typedef std::vector<std::unique_ptr<ConstMatrix> > ConstMatrixVector;

  explicit TensorArrayPackOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    const int32 array_size = tensor_array->Size();
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));

    // Simplest case
    if (array_size == 0) {
      Tensor empty(dtype_, TensorShape({}));
      ctx->set_output(0, empty);
      return;
    }

    PersistentTensor value_0;
    OP_REQUIRES_OK(ctx, tensor_array->Read(0, &value_0));
    Tensor* value_0_t = value_0.AccessTensor(ctx);
    TensorShape output_shape(value_0_t->shape());
    output_shape.InsertDim(0, array_size);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    ConstMatrixVector input_tensors_flat;
    input_tensors_flat.reserve(array_size);
    auto output_flat =
        output_tensor->shaped<T, 2>({1, output_shape.num_elements()});

    for (int i = 0; i < array_size; ++i) {
      PersistentTensor value;
      OP_REQUIRES_OK(ctx, tensor_array->Read(i, &value));
      const Tensor* value_t = value.AccessTensor(ctx);
      OP_REQUIRES(
          ctx, value_0_t->shape() == value_t->shape(),
          errors::InvalidArgument(
              "TensorArray has inconsistent shapes.  Index 0 has shape: ",
              value_0_t->shape().DebugString(), " but index ", i,
              " has shape: ", value_t->shape().DebugString()));
      input_tensors_flat.emplace_back(
          new ConstMatrix(value_t->shaped<T, 2>({1, value_t->NumElements()})));
    }

    if (std::is_same<Device, GPUDevice>::value) {
      ConcatGPU<T>(ctx->eigen_gpu_device(), input_tensors_flat, &output_flat);
    } else {
      ConcatCPU<T>(ctx->device(), input_tensors_flat, &output_flat);
    }
  }

 private:
  DataType dtype_;
};

#define REGISTER_PACK(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")            \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle"),         \
                          TensorArrayPackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_PACK);
REGISTER_PACK(quint8);
REGISTER_PACK(qint8);
REGISTER_PACK(qint32);
REGISTER_PACK(bfloat16);

#undef REGISTER_PACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle"),         \
                          TensorArrayPackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("handle"),
                        TensorArrayPackOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class TensorArrayUnpackOp : public OpKernel {
 public:
  explicit TensorArrayUnpackOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("gradient_add", &gradient_add_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));

    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));

    const int32 array_size = tensor_array->Size();
    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    TensorShape element_shape(tensor_value->shape());
    OP_REQUIRES(ctx, element_shape.dims() > 0,
                errors::InvalidArgument("Input value for unpack must be at "
                                        "least a vector but received shape: ",
                                        element_shape.DebugString()));
    OP_REQUIRES(
        ctx, element_shape.dim_size(0) == array_size,
        errors::InvalidArgument(
            "Input value must have first dimension equal to the array size (",
            element_shape.dim_size(0), " vs. ", array_size, ")"));
    element_shape.RemoveDim(0);

    auto tensor_value_t = tensor_value->shaped<T, 3>(
        {1, array_size, element_shape.num_elements()});

    Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, 0, 0};
    Eigen::DSizes<Eigen::DenseIndex, 3> sizes{1, 1,
                                              element_shape.num_elements()};

    for (int i = 0; i < array_size; ++i) {
      Tensor* tensor_value_i;
      PersistentTensor persistent_tensor;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_persistent(tensor_array->ElemType(), element_shape,
                                        &persistent_tensor, &tensor_value_i));
      auto tensor_value_i_t =
          tensor_value_i->shaped<T, 3>({1, 1, element_shape.num_elements()});
      indices[1] = i;

      functor::Split<Device, T>()(ctx->eigen_device<Device>(), tensor_value_i_t,
                                  tensor_value_t, indices, sizes);

      if (gradient_add_) {
        Status s =
            tensor_array->WriteOrAdd<Device, T>(ctx, i, persistent_tensor);
        OP_REQUIRES_OK(ctx, s);
      } else {
        OP_REQUIRES_OK(ctx, tensor_array->Write(i, persistent_tensor));
      }
    }
  }

 private:
  bool gradient_add_;
};

#define REGISTER_UNPACK(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("TensorArrayUnpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayUnpackOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_UNPACK);
#undef REGISTER_UNPACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayUnpack")      \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("handle"),     \
                          TensorArrayUnpackOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// Delete the TensorArray from its resource container.  This enables
// the user to close and release the resource in the middle of a step/run.
// TODO(ebrevdo): decide whether closing the grad op should happen
// here or on the python side.
class TensorArrayCloseOp : public OpKernel {
 public:
  explicit TensorArrayCloseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    TensorArray* tensor_array;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    // Instead of deleting this TA from the ResourceManager, we just
    // clear it away and mark it as dead.  The remaining memory
    // consumed store its mutex and handle Tensor.  This will be
    // cleared out at the end of the step anyway, so it's fine to keep
    // it around temporarily.  The next call to GetTensorArray will
    // fail because GetTensorArray checks to see if the TensorArray is
    // dead or not.
    tensor_array->ClearAndMarkDead();
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayClose").Device(DEVICE_CPU),
                        TensorArrayCloseOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorArrayClose").Device(DEVICE_GPU).HostMemory("handle"),
    TensorArrayCloseOp);

}  // namespace tensorflow
