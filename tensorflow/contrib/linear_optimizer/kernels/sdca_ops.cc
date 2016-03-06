/* Copyright 2016 Google Inc. All Rights Reserved.

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

// See docs in ../ops/sdca_ops.cc.

#define EIGEN_USE_THREADS

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/linear_optimizer/kernels/logistic-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/resources.h"
#include "tensorflow/contrib/linear_optimizer/kernels/squared-loss.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

// A feature group of a single example by this struct.
struct PerExampleSparseIndicesWeights {
  // N X 1 vector with feature indices.
  Eigen::Tensor</*const*/ int64, 1, Eigen::RowMajor> feature_indices;

  // N X 1 vector with feature values.
  TTypes</*const*/ float>::UnalignedVec feature_values;

  // sum squared norm of the features.
  double norm;
};

struct Regularizations {
  float symmetric_l1 = 0;
  float symmetric_l2 = 0;
};

struct RegularizationLoss {
  double l1_loss = 0;
  double l2_loss = 0;
};

struct PerExampleData {
  double old_wx = 0;
  double wx = 0;
  double norm = 0;
};

// Weights associated with a (sparse or dense) feature group, such that the size
// of WeightsByGroup is the number of feature groups.
using WeightsByGroup = std::vector<TTypes<float>::Vec>;

// SparseExamples represent sparse feature groups of each example.
using SparseExamples =
    std::vector<std::unique_ptr<const PerExampleSparseIndicesWeights>>;

// SparseExamples associated with each sparse feature group.
using SparseExamplesByGroup = std::vector<SparseExamples>;

// Dense features associated with each dense feature group.
using DenseFeaturesByGroup = std::vector<TTypes<const float>::Vec>;

// Go through the entire training set once, in a parallel and partitioned
// fashion, so that we create per-example structures. A non-OK return status
// indicates that the contents of sparse_examples_by_group cannot be trusted or
// used.
Status FillSparseExamplesByGroup(
    const int64 num_sparse_features, const int64 num_examples,
    const OpInputList& sparse_features_indices_inputs,
    const OpInputList& sparse_features_values_inputs,
    const WeightsByGroup& sparse_weights_by_group,
    const DeviceBase::CpuWorkerThreads& worker_threads,
    SparseExamplesByGroup* const sparse_examples_by_group) {
  if (sparse_features_indices_inputs.size() != num_sparse_features ||
      sparse_features_values_inputs.size() != num_sparse_features ||
      sparse_weights_by_group.size() != num_sparse_features) {
    return errors::Internal("Unaligned sparse features.");
  }

  sparse_examples_by_group->clear();
  sparse_examples_by_group->resize(num_sparse_features);

  mutex mu;
  Status result;  // Guarded by mu.
  {
    auto parse_partition = [&](const int64 begin, const int64 end) {
      // We set the order as [0, 1], which specifies that its row-major
      // increasing. This means first column has ids which are lexicographically
      // increasing.
      static const int64 kIndicesDims = 2;
      gtl::InlinedVector<int64, 8> order(kIndicesDims);
      std::iota(order.begin(), order.end(), 0);
      for (int64 i = begin; i < end; ++i) {
        if (sparse_features_indices_inputs[i].shape().dims() != kIndicesDims) {
          mutex_lock l(mu);
          result = errors::InvalidArgument(strings::Printf(
              "Indices should have exactly %lld dimensions. Encountered: %d",
              kIndicesDims, sparse_features_indices_inputs[i].shape().dims()));
          return;
        }

        sparse::SparseTensor st(
            sparse_features_indices_inputs[i], sparse_features_values_inputs[i],
            sparse_features_indices_inputs[i].shape(), order);
        (*sparse_examples_by_group)[i] = SparseExamples(num_examples);
        for (const auto& example_group : st.group({0})) {
          const TTypes<int64>::UnalignedConstMatrix indices =
              example_group.indices();
          const int64 example_index = indices(0, 0);
          if (example_index < 0 || example_index >= num_examples) {
            mutex_lock l(mu);
            result = errors::Internal(strings::Printf(
                "Example indices should be in [0, %lld). Encountered: %lld",
                num_examples, example_index));
            return;
          }

          const auto feature_indices = indices.chip</*dim=*/1>(/*offset=*/1);
          const Eigen::Tensor<int64, 0, Eigen::RowMajor> min_feature_index =
              feature_indices.minimum();
          const Eigen::Tensor<int64, 0, Eigen::RowMajor> max_feature_index =
              feature_indices.maximum();
          if (min_feature_index() < 0 ||
              max_feature_index() >= sparse_weights_by_group[i].size()) {
            mutex_lock l(mu);
            result = errors::InvalidArgument(strings::Printf(
                "Feature indices should be in [0, %ld). Encountered "
                "min:%lld max:%lld for example:%lld",
                sparse_weights_by_group[i].size(), min_feature_index(),
                max_feature_index(), example_index));
            return;
          }

          const Eigen::Tensor<float, 0, Eigen::RowMajor> norm =
              example_group.values<float>().square().sum();
          (*sparse_examples_by_group)[i][example_index].reset(
              new PerExampleSparseIndicesWeights{
                  feature_indices, example_group.values<float>(), norm()});
        }
      }
    };

    // For each column, the cost of parsing it is O(num_examples). We use
    // num_examples here, as empirically Shard() creates the right amount of
    // threads based on the problem size.
    // TODO(rohananil): Tune this as a function of dataset size.
    const int64 kCostPerUnit = num_examples;
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_sparse_features, kCostPerUnit, parse_partition);
  }

  return result;
}

// Compute the shrinkage factor for proximal sdca.
inline double ShrinkageFactor(const Regularizations& regularizations) {
  return regularizations.symmetric_l1 / regularizations.symmetric_l2;
}

// Proximal SDCA shrinking for L1 regularization.
inline double Shrink(const double weight, const double shrink_by) {
  const double shrink_weight = std::max(std::abs(weight) - shrink_by, 0.0);
  if (shrink_weight > 0.0) {
    return std::copysign(shrink_weight, weight);
  }
  return 0.0;
}

// Compute L1 and L2 regularization loss.
inline RegularizationLoss ComputeRegularizationLoss(
    const WeightsByGroup& sparse_weights_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const Regularizations& regularizations) {
  RegularizationLoss result;

  const double shrink_by = ShrinkageFactor(regularizations);
  auto accumulate_regularization_loss = [&](const double w) {
    const double sw = std::abs(Shrink(w, shrink_by));
    result.l1_loss += sw;
    result.l2_loss += sw * sw;
  };

  for (const TTypes<float>::Vec weights : sparse_weights_by_group) {
    for (int64 i = 0; i < weights.size(); ++i) {
      accumulate_regularization_loss(weights(i));
    }
  }

  for (const TTypes<float>::Vec weights : dense_weights_by_group) {
    accumulate_regularization_loss(weights(0));
  }

  result.l1_loss *= regularizations.symmetric_l1;
  result.l2_loss *= regularizations.symmetric_l2;
  return result;
}

// Compute PerExampleData which contains the logits, and weighted example norm
// for a given example_id. Norm is weighted by 1/(lambda*N).
inline PerExampleData ComputeWxAndWeightedExampleNorm(
    const int64 example_id,  //
    const WeightsByGroup& sparse_weights_by_group,
    const WeightsByGroup& sparse_delta_weights_by_group,
    const SparseExamplesByGroup& sparse_examples_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const WeightsByGroup& dense_delta_weights_by_group,
    const DenseFeaturesByGroup& dense_features_by_group,
    const Regularizations& regularizations) {
  PerExampleData result;
  const double shrink_by = ShrinkageFactor(regularizations);
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_indices_values = sparse_examples_by_group[i];
    const TTypes<float>::Vec weights = sparse_weights_by_group[i];
    const TTypes<float>::Vec delta_weights = sparse_delta_weights_by_group[i];
    if (sparse_indices_values[example_id]) {
      const auto indices = sparse_indices_values[example_id]->feature_indices;
      const auto values = sparse_indices_values[example_id]->feature_values;
      for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
        const int64 index = indices(dim);
        const double weight = weights(index);
        const double value = values(dim);
        result.old_wx += Shrink(weight, shrink_by) * value;
        result.wx += Shrink(weight + delta_weights(index), shrink_by) * value;
      }
      result.norm += sparse_indices_values[example_id]->norm;
    }
  }
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    const double weight = dense_weights_by_group[i](0);
    const double value = dense_features_by_group[i](example_id);
    result.old_wx += Shrink(weight, shrink_by) * value;
    result.wx +=
        Shrink(weight + dense_delta_weights_by_group[i](0), shrink_by) * value;
    result.norm += value * value;
  }
  result.norm /= regularizations.symmetric_l2;
  return result;
}

// Zeros out all the weights.
void SetZeroDeltaWeights(WeightsByGroup* const sparse_delta_weights_by_group,
                         WeightsByGroup* const dense_delta_weights_by_group) {
  // TODO(rohananil): Parallelize this.
  for (TTypes<float>::Vec delta_weights : *sparse_delta_weights_by_group) {
    delta_weights.setZero();
  }
  for (TTypes<float>::Vec delta_weights : *dense_delta_weights_by_group) {
    delta_weights.setZero();
  }
}

// Add delta weights to original weights.
void AddDeltaWeights(const WeightsByGroup& src, WeightsByGroup* const dst) {
  // TODO(rohananil): Parallelize this.
  for (size_t group = 0; group < src.size(); ++group) {
    for (size_t i = 0; i < src[group].size(); ++i) {
      (*dst)[group](i) += src[group](i);
    }
  }
}

// Apply L1 regularization on the weights,
void ShrinkWeights(const Regularizations& regularizations,
                   WeightsByGroup* const sparse_weights_by_group,
                   WeightsByGroup* const dense_weights_by_group) {
  const double shrink_by = ShrinkageFactor(regularizations);
  for (TTypes<float>::Vec weights : *sparse_weights_by_group) {
    for (int64 i = 0; i < weights.size(); ++i) {
      weights(i) = Shrink(weights(i), shrink_by);
    }
  }
  for (TTypes<float>::Vec weights : *dense_weights_by_group) {
    weights(0) = Shrink(weights(0), shrink_by);
  }
}

void UpdateDeltaWeights(const int64 example_id,
                        const SparseExamplesByGroup& sparse_examples_by_group,
                        const DenseFeaturesByGroup& dense_features_by_group,
                        const double bounded_dual_delta,
                        const double l2_regularization,
                        WeightsByGroup* const sparse_delta_weights_by_group,
                        WeightsByGroup* const dense_delta_weights_by_group) {
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_examples = sparse_examples_by_group[i];
    TTypes<float>::Vec delta_weights = (*sparse_delta_weights_by_group)[i];
    if (sparse_examples[example_id]) {
      const auto indices = sparse_examples[example_id]->feature_indices;
      const auto values = sparse_examples[example_id]->feature_values;
      for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
        // TODO(rohananil): Atomic updates provide better convergence guarantees
        // However, casting float to atomic<float> is UB. We may consider
        // sharded set of locks, or bring primal-dual relationship to consistent
        // state after several epochs.
        delta_weights(indices(dim)) +=
            bounded_dual_delta * values(dim) / l2_regularization;
      }
    }
  }
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    const auto values = dense_features_by_group[i];
    TTypes<float>::Vec delta_weights = (*dense_delta_weights_by_group)[i];
    // TODO(rohananil): Atomic updates provide better convergence guarantees
    // However, casting float to atomic<float> is UB. We may consider
    // sharded set of locks, or bring primal-dual relationship to consistent
    // state after several epochs.
    delta_weights(0) +=
        bounded_dual_delta * values(example_id) / l2_regularization;
  }
}

// Atomically add a double to a std::atomic<double>.
inline void AtomicAdd(const double src, std::atomic<double>* const dst) {
  // We use a strong version of compare-exchange, as weak version can spuriously
  // fail.
  for (double c = dst->load(); !dst->compare_exchange_strong(c, c + src);) {
  }
}

WeightsByGroup MakeWeightsFrom(OpMutableInputList* const input_list) {
  WeightsByGroup result;
  for (int i = 0; i < input_list->size(); ++i) {
    result.emplace_back(input_list->at(i, /*lock_held=*/true).flat<float>());
  }
  return result;
}

std::vector<Tensor> MakeTensorsLike(OpMutableInputList* const input_list) {
  std::vector<Tensor> result;
  for (int i = 0; i < input_list->size(); ++i) {
    result.emplace_back(DT_FLOAT,
                        input_list->at(i, /*lock_held=*/true).shape());
    result.back().flat<float>().setZero();
  }
  return result;
}

WeightsByGroup MakeDeltaWeightsFrom(std::vector<Tensor>* const tensors) {
  WeightsByGroup result;
  for (auto& tensor : *tensors) {
    result.emplace_back(tensor.flat<float>());
  }
  return result;
}

}  // namespace

class SdcaSolver : public OpKernel {
 public:
  explicit SdcaSolver(OpKernelConstruction* context) : OpKernel(context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      compute_dual_loss_ = logistic_loss::ComputeDualLoss;
      compute_primal_loss_ = logistic_loss::ComputePrimalLoss;
      compute_dual_update_ = logistic_loss::ComputeUpdatedDual;
      convert_label_ = logistic_loss::ConvertLabel;
    } else if (loss_type == "squared_loss") {
      compute_dual_loss_ = squared_loss::ComputeDualLoss;
      compute_primal_loss_ = squared_loss::ComputePrimalLoss;
      compute_dual_update_ = squared_loss::ComputeUpdatedDual;
      convert_label_ = squared_loss::ConvertLabel;
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }
    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features",
                                             &num_sparse_features_));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_dense_features", &num_dense_features_));
    OP_REQUIRES(
        context, num_sparse_features_ + num_dense_features_ > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("l1", &regularizations_.symmetric_l1));
    OP_REQUIRES_OK(context,
                   context->GetAttr("l2", &regularizations_.symmetric_l2));
    // We enforce a minimal l2, required by the algorithm.
    regularizations_.symmetric_l2 =
        std::max(regularizations_.symmetric_l2, 1.0f);

    OP_REQUIRES_OK(context, context->GetAttr("duality_gap_threshold",
                                             &duality_gap_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
    OP_REQUIRES_OK(context, context->GetAttr("solver_uuid", &solver_uuid_));
  }

  void Compute(OpKernelContext* context) override {
    // Get a handle on a shared container across invocations of this Kernel.
    // The shared container is intended to maintain state at the example level
    // across invocations of the kernel on different input data.
    //
    // TODO(katsiapis): Replace this in-Kernel data structure with a first class
    // citizen mutable Dictionary in tensorflow proper, that we will initialize
    // and update externally.
    DataByExample* data_by_example = nullptr;
    OP_REQUIRES_OK(context,
                   context->resource_manager()->LookupOrCreate<DataByExample>(
                       container_, solver_uuid_, &data_by_example,
                       [this](DataByExample** ret) {
                         *ret = new DataByExample(container_, solver_uuid_);
                         return Status::OK();
                       }));
    OP_REQUIRES(
        context, !data_by_example->RefCountIsOne(),
        errors::Internal("Expected shared-ownership of duals_by_example."));

    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input("example_weights", &example_weights_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_weights_t->shape()),
                errors::InvalidArgument("example_weights should be a vector."));
    const auto example_weights = example_weights_t->vec<float>();

    Eigen::Tensor<float, 0, Eigen::RowMajor> example_weights_sum;
    example_weights_sum.device(context->eigen_cpu_device()) =
        example_weights.sum();
    const float weighted_examples = example_weights_sum();
    const int64 num_examples = example_weights.size();

    OP_REQUIRES(context, weighted_examples > 0,
                errors::InvalidArgument("No weighted examples in ",
                                        num_examples, " training examples"));

    Tensor primal_loss_t;
    OP_REQUIRES_OK(context,
                   context->mutable_input("primal_loss", &primal_loss_t,
                                          /*lock_held=*/true));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(primal_loss_t.shape()),
                errors::InvalidArgument("primal_loss should be a scalar."));
    auto primal_loss = primal_loss_t.scalar<double>();

    OpInputList dense_features_inputs;
    OP_REQUIRES_OK(
        context, context->input_list("dense_features", &dense_features_inputs));

    DenseFeaturesByGroup dense_features_by_group;
    for (const auto& dense_feature : dense_features_inputs) {
      dense_features_by_group.emplace_back(dense_feature.vec<float>());
    }

    const Tensor* example_labels_t;
    OP_REQUIRES_OK(context,
                   context->input("example_labels", &example_labels_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_labels_t->shape()),
                errors::InvalidArgument("example_labels should be a vector."));
    const auto example_labels = example_labels_t->vec<float>();
    OP_REQUIRES(context, example_labels.size() == num_examples,
                errors::InvalidArgument(strings::Printf(
                    "The number of example labels (%ld) should match the "
                    "number of example weights (%lld).",
                    example_labels.size(), num_examples)));

    const Tensor* example_ids_t;
    OP_REQUIRES_OK(context, context->input("example_ids", &example_ids_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_ids_t->shape()),
                errors::InvalidArgument("example_ids should be a vector."));
    const auto example_ids = example_ids_t->vec<string>();
    OP_REQUIRES(context, example_labels.size() == num_examples,
                errors::InvalidArgument(strings::Printf(
                    "The number of example ids (%ld) should match the number "
                    "of example weights (%lld).",
                    example_ids.size(), num_examples)));
    const int64 num_duplicate_example_ids = [&] {
      // TODO(katsiapis): Benchmark and/or optimize.
      std::vector<StringPiece> scratch_storage;
      scratch_storage.reserve(example_ids.size());
      for (size_t i = 0; i < example_ids.size(); ++i) {
        scratch_storage.emplace_back(example_ids(i));
      }
      std::sort(scratch_storage.begin(), scratch_storage.end());
      return std::distance(
          std::unique(scratch_storage.begin(), scratch_storage.end()),
          scratch_storage.end());
    }();
    OP_REQUIRES(context, num_duplicate_example_ids == 0,
                errors::InvalidArgument(strings::Printf(
                    "Detected %lld duplicates in example_ids, which usually "
                    "indicates a bug in the input data.",
                    num_duplicate_example_ids)));

    OpMutableInputList sparse_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list(
                                "sparse_weights", &sparse_weights_inputs));
    WeightsByGroup sparse_weights_by_group =
        MakeWeightsFrom(&sparse_weights_inputs);
    std::vector<Tensor> sparse_delta_weights_by_group_backing_store =
        MakeTensorsLike(&sparse_weights_inputs);
    WeightsByGroup sparse_delta_weights_by_group =
        MakeDeltaWeightsFrom(&sparse_delta_weights_by_group_backing_store);

    // TODO(rohananil): Remove the code duplication between sparse and
    // dense weights.
    OpMutableInputList dense_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list("dense_weights",
                                                        &dense_weights_inputs));
    WeightsByGroup dense_weights_by_group =
        MakeWeightsFrom(&dense_weights_inputs);
    std::vector<Tensor> dense_delta_weights_by_group_backing_store =
        MakeTensorsLike(&dense_weights_inputs);
    WeightsByGroup dense_delta_weights_by_group =
        MakeDeltaWeightsFrom(&dense_delta_weights_by_group_backing_store);

    OpInputList sparse_features_indices_inputs;
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_features_indices",
                                       &sparse_features_indices_inputs));
    OpInputList sparse_features_values_inputs;
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_features_values",
                                       &sparse_features_values_inputs));
    SparseExamplesByGroup sparse_examples_by_group;
    OP_REQUIRES_OK(
        context,
        FillSparseExamplesByGroup(
            num_sparse_features_, num_examples, sparse_features_indices_inputs,
            sparse_features_values_inputs, sparse_weights_by_group,
            *context->device()->tensorflow_cpu_worker_threads(),
            &sparse_examples_by_group));

    // Those will be shuffled below at each iteration and processed in a
    // partitioned fashion across multiple threads.
    std::vector<int64> example_indices(num_examples);
    std::iota(example_indices.begin(), example_indices.end(), 0);

    std::random_device random_device;
    std::mt19937 random_generator(random_device());

    // Break when duality gap |P(w) - D(alpha)| is less than
    // duality_gap_threshold_
    double total_duality_gap = std::numeric_limits<double>::max();
    while ((total_duality_gap / weighted_examples) > duality_gap_threshold_) {
      std::atomic<double> total_primal_loss(0);
      std::atomic<double> total_dual_loss(0);
      SetZeroDeltaWeights(&sparse_delta_weights_by_group,
                          &dense_delta_weights_by_group);

      // Compute regularization loss at the start of the iteration so that
      // we can compute an exact value of duality gap (for the weights from
      // the previous iteration).
      const RegularizationLoss regularization_loss = ComputeRegularizationLoss(
          sparse_weights_by_group, dense_weights_by_group, regularizations_);

      // Randomize the examples across iterations for faster convergence.
      std::shuffle(example_indices.begin(), example_indices.end(),
                   random_generator);

      {
        // Process examples in parallel, in a partitioned fashion.
        mutex mu;
        Status update_status;  // Guarded by mu.
        auto update_partition = [&](const int64 begin, const int64 end) {
          double dual_loss_on_example_subset = 0;
          double primal_loss_on_example_subset = 0;
          for (int64 offset = begin; offset < end; ++offset) {
            // Get example id, label, and weight.
            const int64 example_index = example_indices[offset];
            const DataByExample::Key example_key =
                DataByExample::MakeKey(example_ids(example_index));
            const double current_dual = (*data_by_example)[example_key].dual;
            const double example_weight = example_weights(example_index);
            float example_label = example_labels(example_index);
            const Status conversion_status = convert_label_(&example_label);
            if (!conversion_status.ok()) {
              mutex_lock l(mu);
              update_status = conversion_status;
              // Return from this worker thread - the calling thread is
              // responsible for checking context status and returning on error.
              return;
            }

            // Compute wx, example norm weighted by regularization, dual loss,
            // primal loss.
            const PerExampleData per_example_data =
                ComputeWxAndWeightedExampleNorm(
                    example_index, sparse_weights_by_group,
                    sparse_delta_weights_by_group, sparse_examples_by_group,
                    dense_weights_by_group, dense_delta_weights_by_group,
                    dense_features_by_group, regularizations_);
            // Compute primal based on the previous iteration.
            primal_loss_on_example_subset += compute_primal_loss_(
                per_example_data.old_wx, example_label, example_weight);

            const double primal_loss = compute_primal_loss_(
                per_example_data.wx, example_label, example_weight);

            const double dual_loss =
                compute_dual_loss_(current_dual, example_label, example_weight);
            dual_loss_on_example_subset += dual_loss;

            const double new_dual = compute_dual_update_(
                example_label, example_weight, current_dual,
                per_example_data.wx, per_example_data.norm, primal_loss,
                dual_loss);

            // Compute new weights.
            const double bounded_dual_delta =
                (new_dual - current_dual) * example_weight;
            UpdateDeltaWeights(example_index, sparse_examples_by_group,
                               dense_features_by_group, bounded_dual_delta,
                               regularizations_.symmetric_l2,
                               &sparse_delta_weights_by_group,
                               &dense_delta_weights_by_group);

            // Update dual variable.
            (*data_by_example)[example_key].dual = new_dual;
          }
          AtomicAdd(primal_loss_on_example_subset, &total_primal_loss);
          AtomicAdd(dual_loss_on_example_subset, &total_dual_loss);
          // TODO(rohananil): We may in the future want to make the primal-dual
          // relationship consistent as our current updates are not
          // transactional.
        };
        const DeviceBase::CpuWorkerThreads& worker_threads =
            *context->device()->tensorflow_cpu_worker_threads();
        // TODO(katsiapis): Current multiplier (100,000) works well empirically
        // but perhaps we can tune it better.
        const int64 kCostPerUnit =
            100000 * (num_sparse_features_ + num_dense_features_);
        Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
              kCostPerUnit, update_partition);
        OP_REQUIRES_OK(context, update_status);
      }

      total_duality_gap = total_primal_loss.load() + total_dual_loss.load() +
                          regularization_loss.l1_loss +
                          regularization_loss.l2_loss;
      primal_loss() = (total_primal_loss.load() + regularization_loss.l1_loss +
                       regularization_loss.l2_loss) /
                      weighted_examples;
      AddDeltaWeights(sparse_delta_weights_by_group, &sparse_weights_by_group);
      AddDeltaWeights(dense_delta_weights_by_group, &dense_weights_by_group);
    }
    ShrinkWeights(regularizations_, &sparse_weights_by_group,
                  &dense_weights_by_group);

    // TODO(katsiapis): Use core::ScopedUnref once it's moved out of internal.
    data_by_example->Unref();
  }

 private:
  std::function<decltype(logistic_loss::ComputeDualLoss)> compute_dual_loss_;
  std::function<decltype(logistic_loss::ComputePrimalLoss)>
      compute_primal_loss_;
  std::function<decltype(logistic_loss::ComputeUpdatedDual)>
      compute_dual_update_;
  std::function<decltype(logistic_loss::ConvertLabel)> convert_label_;
  int64 num_sparse_features_;
  int64 num_dense_features_;
  Regularizations regularizations_;
  float duality_gap_threshold_;
  string container_;
  string solver_uuid_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaSolver").Device(DEVICE_CPU), SdcaSolver);

}  // namespace tensorflow
