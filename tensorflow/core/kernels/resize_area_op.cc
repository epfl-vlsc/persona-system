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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <algorithm>
#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeAreaOp : public OpKernel {
 public:
  explicit ResizeAreaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto Svec = shape_t.vec<int32>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({input.dim_size(0), Svec(0),
                                                Svec(1), input.dim_size(3)}),
                                &output));
    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);
    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    // A temporary tensor for computing the sum.
    Tensor sum_tensor;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({channels}), &sum_tensor));
    typename TTypes<float, 1>::Tensor sum_data = sum_tensor.vec<float>();

    const float height_scale =
        (align_corners_ && out_height > 1)
            ? (in_height - 1) / static_cast<float>(out_height - 1)
            : in_height / static_cast<float>(out_height);
    const float width_scale =
        (align_corners_ && out_width > 1)
            ? (in_width - 1) / static_cast<float>(out_width - 1)
            : in_width / static_cast<float>(out_width);

    // When using this algorithm for downsizing, the target pixel value is the
    // weighted average of all the source pixels. The weight is determined by
    // the contribution percentage of the source pixel.
    //
    // Let "scale" be "target_image_size/source_image_size". If 1/n of the
    // source pixel contributes to the target pixel, then the weight is (1/n *
    // scale); if the complete source pixel contributes to the target pixel,
    // then the weight is scale.
    //
    // To visualize the implementation, use one dimension as an example:
    // Resize in[4] to out[3].
    //   scale = 3/4 = 0.75
    //   out[0]: in[0] and 1/3 of in[1]
    //   out[1]: 2/3 of in[1] and 2/3 of in[2]
    //   out[2]: 1/3 of in[2] and in[1]
    // Hence, the output pixel values are:
    //   out[0] = (in[0] * 1.0 + in[1] * 1/3) * scale
    //   out[1] = (in[1] * 2/3 + in[2] * 2/3 * scale
    //   out[2] = (in[3] * 1/3 + in[3] * 1.0) * scale
    float scale = 1.0 / (height_scale * width_scale);
    for (int64 b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const float in_y = y * height_scale;
        const float in_y1 = (y + 1) * height_scale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        int64 y_start = floor(in_y);
        int64 y_end = ceil(in_y1);

        for (int64 x = 0; x < out_width; ++x) {
          const float in_x = x * width_scale;
          const float in_x1 = (x + 1) * width_scale;
          // The start and end width indices of all the cells that could
          // contribute to the target cell.
          int64 x_start = floor(in_x);
          int64 x_end = ceil(in_x1);

          sum_data.setConstant(0.0);
          for (int64 i = y_start; i < y_end; ++i) {
            float scale_y =
                i < in_y ? i + 1 - in_y : (i + 1 > in_y1 ? in_y1 - i : 1.0);
            for (int64 j = x_start; j < x_end; ++j) {
              float scale_x =
                  j < in_x ? j + 1 - in_x : (j + 1 > in_x1 ? in_x1 - j : 1.0);
              for (int64 c = 0; c < channels; ++c) {
#define BOUND(val, limit) std::min(((limit)-1ll), (std::max(0ll, (val))))
                sum_data(c) +=
                    input_data(b, BOUND(i, in_height), BOUND(j, in_width), c) *
                    scale_y * scale_x * scale;
#undef BOUND
              }
            }
          }
          for (int64 c = 0; c < channels; ++c) {
            output_data(b, y, x, c) = sum_data(c);
          }
        }
      }
    }
  }

 private:
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeArea")          \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeAreaOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // namespace tensorflow
