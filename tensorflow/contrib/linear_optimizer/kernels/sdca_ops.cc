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
#include <string>
#include <unordered_set>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/linear_optimizer/kernels/hinge-loss.h"
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
#include "tensorflow/core/kernels/bounds_check.h"
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

struct PerExampleData {
  double wx = 0;
  double norm = 0;
};

// Weights associated with a (sparse or dense) feature group, such that the size
// of WeightsByGroup is the number of feature groups.
using WeightsByGroup = std::vector<TTypes<float>::Vec>;

// DeltaWeights associated with a (sparse or dense) feature group, such that the
// size of DeltaWeightsByGroup is the number of feature groups. Atomicity is
// required when changing the weights in order to have transactional updates.
using DeltaWeightsByGroup = std::vector<std::vector<std::atomic<double>>>;

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
    const int64 num_sparse_features, const int num_examples,
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
  Status result GUARDED_BY(mu);
  {
    auto parse_partition = [&](const int64 begin, const int64 end) {
      // We set the order as [0, 1], which specifies that its row-major
      // increasing. This means first column has ids which are lexicographically
      // increasing.
      static const int64 kIndicesDims = 2;
      gtl::InlinedVector<int64, 8> order(kIndicesDims);
      std::iota(order.begin(), order.end(), 0);

      // The static_cast here is safe since begin and end can be at most
      // num_examples which is an int.
      for (int i = static_cast<int>(begin); i < end; ++i) {
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
          const int64 example_index = internal::SubtleMustCopy(indices(0, 0));
          if (example_index < 0 || example_index >= num_examples) {
            mutex_lock l(mu);
            result = errors::Internal(strings::Printf(
                "Example indices should be in [0, %d). Encountered: %lld",
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
    // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
    const int64 kCostPerUnit = num_examples;
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_sparse_features, kCostPerUnit, parse_partition);
  }

  return result;
}

// Atomically add a double to a std::atomic<double>.
inline void AtomicAdd(const double src, std::atomic<double>* const dst) {
  // We use a strong version of compare-exchange, as weak version can spuriously
  // fail.
  for (double c = dst->load(); !dst->compare_exchange_strong(c, c + src);) {
  }
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

// Compute PerExampleData which contains the logits, and weighted example norm
// for a given example_id. Norm is weighted by 1/(lambda*N).
inline PerExampleData ComputeWxAndWeightedExampleNorm(
    const int64 example_id,  //
    const WeightsByGroup& sparse_weights_by_group,
    const DeltaWeightsByGroup& sparse_delta_weights_by_group,
    const SparseExamplesByGroup& sparse_examples_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const DeltaWeightsByGroup& dense_delta_weights_by_group,
    const DenseFeaturesByGroup& dense_features_by_group,
    const Regularizations& regularizations) {
  PerExampleData result;
  const double shrink_by = ShrinkageFactor(regularizations);
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_indices_values = sparse_examples_by_group[i];
    const TTypes<float>::Vec weights = sparse_weights_by_group[i];
    const std::vector<std::atomic<double>>& delta_weights =
        sparse_delta_weights_by_group[i];
    if (sparse_indices_values[example_id]) {
      const auto indices = sparse_indices_values[example_id]->feature_indices;
      const auto values = sparse_indices_values[example_id]->feature_values;
      for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
        const int64 index = internal::SubtleMustCopy(indices(dim));
        const double weight = weights(index);
        const double value = values(dim);
        result.wx +=
            Shrink(weight + delta_weights[index].load(), shrink_by) * value;
      }
      result.norm += sparse_indices_values[example_id]->norm;
    }
  }
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    // (0) and [0] access guaranteed to be ok due to ValidateDenseWeights().
    const double weight = dense_weights_by_group[i](0);
    const double value = dense_features_by_group[i](example_id);
    result.wx +=
        Shrink(weight + dense_delta_weights_by_group[i][0].load(), shrink_by) *
        value;
    result.norm += value * value;
  }
  result.norm /= regularizations.symmetric_l2;
  return result;
}

// Add delta weights to original weights.
void AddDeltaWeights(const DeltaWeightsByGroup& src,
                     WeightsByGroup* const dst) {
  // TODO(sibyl-Aix6ihai): Parallelize this.
  for (size_t group = 0; group < src.size(); ++group) {
    for (size_t i = 0; i < src[group].size(); ++i) {
      (*dst)[group](i) += src[group][i].load();
    }
  }
}

void UpdateDeltaWeights(
    const int64 example_id,
    const SparseExamplesByGroup& sparse_examples_by_group,
    const DenseFeaturesByGroup& dense_features_by_group,
    const double bounded_dual_delta, const double l2_regularization,
    DeltaWeightsByGroup* const sparse_delta_weights_by_group,
    DeltaWeightsByGroup* const dense_delta_weights_by_group) {
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_examples = sparse_examples_by_group[i];
    std::vector<std::atomic<double>>& delta_weights =
        (*sparse_delta_weights_by_group)[i];
    if (sparse_examples[example_id]) {
      const auto indices = sparse_examples[example_id]->feature_indices;
      const auto values = sparse_examples[example_id]->feature_values;
      for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
        AtomicAdd(bounded_dual_delta * values(dim) / l2_regularization,
                  &delta_weights[indices(dim)]);
      }
    }
  }
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    const auto values = dense_features_by_group[i];
    std::vector<std::atomic<double>>& delta_weights =
        (*dense_delta_weights_by_group)[i];
    AtomicAdd(bounded_dual_delta * values(example_id) / l2_regularization,
              // [0] access guaranteed to be ok due to ValidateDenseWeights().
              &delta_weights[0]);
  }
}

WeightsByGroup MakeWeightsFrom(OpMutableInputList* const input_list) {
  WeightsByGroup result;
  for (int i = 0; i < input_list->size(); ++i) {
    result.emplace_back(input_list->at(i, /*lock_held=*/true).flat<float>());
  }
  return result;
}

DeltaWeightsByGroup MakeZeroDeltaWeightsLike(
    const WeightsByGroup& weights_by_group) {
  // TODO(sibyl-Mooth6ku): Maybe parallelize this.
  DeltaWeightsByGroup result;
  for (const TTypes<float>::Vec weights : weights_by_group) {
    result.emplace_back(weights.size());
    std::fill(result.back().begin(), result.back().end(), 0);
  }
  return result;
}

Status RunTrainStepsForMiniBatch(
    const int num_examples, const TTypes<const string>::Vec example_ids,
    const TTypes<const float>::Vec example_labels,
    const TTypes<const float>::Vec example_weights,
    const DeviceBase::CpuWorkerThreads& worker_threads,
    const Regularizations& regularizations,
    const WeightsByGroup& sparse_weights_by_group,
    const SparseExamplesByGroup& sparse_examples_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const DenseFeaturesByGroup& dense_features_by_group,
    const DualLossUpdater& loss_updater,
    DeltaWeightsByGroup* const sparse_delta_weights_by_group,
    DeltaWeightsByGroup* const dense_delta_weights_by_group,
    DataByExample* const data_by_example) {
  // Process examples in parallel, in a partitioned fashion.
  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  auto train_step = [&](const int64 begin, const int64 end) {
    for (int64 example_index = begin; example_index < end; ++example_index) {
      // Get example id, label, and weight.
      const DataByExample::Key example_key =
          DataByExample::MakeKey(example_ids(example_index));
      DataByExample::Data data = data_by_example->Get(example_key);
      const float example_weight = example_weights(example_index);
      float example_label = example_labels(example_index);
      const Status conversion_status =
          loss_updater.ConvertLabel(&example_label);
      if (!conversion_status.ok()) {
        mutex_lock l(mu);
        train_step_status = conversion_status;
        // Return from this worker thread - the calling thread is
        // responsible for checking context status and returning on error.
        return;
      }

      // Compute wx, example norm weighted by regularization, dual loss,
      // primal loss.
      const PerExampleData per_example_data = ComputeWxAndWeightedExampleNorm(
          example_index, sparse_weights_by_group,
          *sparse_delta_weights_by_group, sparse_examples_by_group,
          dense_weights_by_group, *dense_delta_weights_by_group,
          dense_features_by_group, regularizations);

      const double primal_loss = loss_updater.ComputePrimalLoss(
          per_example_data.wx, example_label, example_weight);

      const double dual_loss = loss_updater.ComputeDualLoss(
          data.dual, example_label, example_weight);

      const double new_dual = loss_updater.ComputeUpdatedDual(
          example_label, example_weight, data.dual, per_example_data.wx,
          per_example_data.norm, primal_loss, dual_loss);

      // Compute new weights.
      const double bounded_dual_delta = (new_dual - data.dual) * example_weight;
      UpdateDeltaWeights(
          example_index, sparse_examples_by_group, dense_features_by_group,
          bounded_dual_delta, regularizations.symmetric_l2,
          sparse_delta_weights_by_group, dense_delta_weights_by_group);

      // Update example data.
      data.dual = new_dual;
      data.primal_loss = primal_loss;
      data.dual_loss = dual_loss;
      data.example_weight = example_weight;
      data_by_example->Set(example_key, data);
    }
  };
  // TODO(sibyl-Aix6ihai): Current multiplier 100000 works well empirically
  // but perhaps we can tune it better.
  const int64 kCostPerUnit = 100000 * (sparse_examples_by_group.size() +
                                       dense_features_by_group.size());
  Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
        kCostPerUnit, train_step);
  return train_step_status;
}

Status FillRegularizations(OpKernelConstruction* const context,
                           Regularizations* const regularizations) {
  TF_RETURN_IF_ERROR(context->GetAttr("l1", &regularizations->symmetric_l1));
  TF_RETURN_IF_ERROR(context->GetAttr("l2", &regularizations->symmetric_l2));
  return Status::OK();
}

// TODO(sibyl-Mooth6ku): Support arbitrary dimensional dense weights and remove this.
Status ValidateDenseWeights(const WeightsByGroup& weights_by_group) {
  for (const TTypes<float>::Vec weights : weights_by_group) {
    if (weights.size() != 1) {
      return errors::InvalidArgument(strings::Printf(
          "Dense weight vectors should have exactly one entry. Found (%ld). "
          "This is probably due to a misconfiguration in the optimizer setup.",
          weights.size()));
    }
  }
  return Status::OK();
}

}  // namespace

class SdcaSolver : public OpKernel {
 public:
  explicit SdcaSolver(OpKernelConstruction* context) : OpKernel(context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater_.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater_.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater_.reset(new HingeLossUpdater);
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
    OP_REQUIRES_OK(context, FillRegularizations(context, &regularizations_));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations_));
    OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
    OP_REQUIRES_OK(context, context->GetAttr("solver_uuid", &solver_uuid_));
  }

  void Compute(OpKernelContext* context) override {
    // Get a handle on a shared container across invocations of this Kernel.
    // The shared container is intended to maintain state at the example level
    // across invocations of the kernel on different input data.
    //
    // TODO(sibyl-Mooth6ku): Replace this in-Kernel data structure with a first class
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
        errors::Internal("Expected shared-ownership of data_by_example."));

    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input("example_weights", &example_weights_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_weights_t->shape()),
                errors::InvalidArgument("example_weights should be a vector."));
    const auto example_weights = example_weights_t->vec<float>();
    OP_REQUIRES(context,
                example_weights.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(strings::Printf(
                    "Too many examples in a mini-batch: %ld > %d",
                    example_weights.size(), std::numeric_limits<int>::max())));
    const int num_examples = static_cast<int>(example_weights.size());

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
                    "number of example weights (%d).",
                    example_labels.size(), num_examples)));

    const Tensor* example_ids_t;
    OP_REQUIRES_OK(context, context->input("example_ids", &example_ids_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_ids_t->shape()),
                errors::InvalidArgument("example_ids should be a vector."));
    const auto example_ids = example_ids_t->vec<string>();
    OP_REQUIRES(context, example_labels.size() == num_examples,
                errors::InvalidArgument(strings::Printf(
                    "The number of example ids (%ld) should match the number "
                    "of example weights (%d).",
                    example_ids.size(), num_examples)));

    OpMutableInputList sparse_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list(
                                "sparse_weights", &sparse_weights_inputs));
    WeightsByGroup sparse_weights_by_group =
        MakeWeightsFrom(&sparse_weights_inputs);
    DeltaWeightsByGroup sparse_delta_weights_by_group =
        MakeZeroDeltaWeightsLike(sparse_weights_by_group);

    OpMutableInputList dense_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list("dense_weights",
                                                        &dense_weights_inputs));
    WeightsByGroup dense_weights_by_group =
        MakeWeightsFrom(&dense_weights_inputs);
    OP_REQUIRES_OK(context, ValidateDenseWeights(dense_weights_by_group));
    DeltaWeightsByGroup dense_delta_weights_by_group =
        MakeZeroDeltaWeightsLike(dense_weights_by_group);

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

    for (int i = 0; i < num_inner_iterations_; ++i) {
      OP_REQUIRES_OK(
          context,
          RunTrainStepsForMiniBatch(
              num_examples, example_ids, example_labels, example_weights,
              *context->device()->tensorflow_cpu_worker_threads(),
              regularizations_, sparse_weights_by_group,
              sparse_examples_by_group, dense_weights_by_group,
              dense_features_by_group, *loss_updater_,
              &sparse_delta_weights_by_group, &dense_delta_weights_by_group,
              data_by_example));
    }
    AddDeltaWeights(sparse_delta_weights_by_group, &sparse_weights_by_group);
    AddDeltaWeights(dense_delta_weights_by_group, &dense_weights_by_group);

    // TODO(sibyl-Mooth6ku): Use core::ScopedUnref once it's moved out of internal.
    data_by_example->Unref();
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  std::unique_ptr<DualLossUpdater> loss_updater_;
  int64 num_sparse_features_;
  int64 num_dense_features_;
  Regularizations regularizations_;
  int num_inner_iterations_;
  string container_;
  string solver_uuid_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaSolver").Device(DEVICE_CPU), SdcaSolver);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, FillRegularizations(context, &regularizations_));
  }

  void Compute(OpKernelContext* context) override {
    OpMutableInputList sparse_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list(
                                "sparse_weights", &sparse_weights_inputs));
    WeightsByGroup sparse_weights_by_group =
        MakeWeightsFrom(&sparse_weights_inputs);

    OpMutableInputList dense_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list("dense_weights",
                                                        &dense_weights_inputs));
    WeightsByGroup dense_weights_by_group =
        MakeWeightsFrom(&dense_weights_inputs);
    OP_REQUIRES_OK(context, ValidateDenseWeights(dense_weights_by_group));

    // TODO(sibyl-Aix6ihai): Parallelize this.
    const double shrink_by = ShrinkageFactor(regularizations_);
    for (TTypes<float>::Vec weights : sparse_weights_by_group) {
      for (int64 i = 0; i < weights.size(); ++i) {
        weights(i) = Shrink(weights(i), shrink_by);
      }
    }
    for (TTypes<float>::Vec weights : dense_weights_by_group) {
      // (0) access guaranteed to be ok due to ValidateDenseWeights().
      weights(0) = Shrink(weights(0), shrink_by);
    }
  }

 private:
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaShrinkL1").Device(DEVICE_CPU), SdcaShrinkL1);

class SdcaTrainingStats : public OpKernel {
 public:
  explicit SdcaTrainingStats(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
    OP_REQUIRES_OK(context, context->GetAttr("solver_uuid", &solver_uuid_));
  }

  void Compute(OpKernelContext* context) override {
    DataByExample* data_by_example = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup<DataByExample>(
                                container_, solver_uuid_, &data_by_example));
    OP_REQUIRES(
        context, !data_by_example->RefCountIsOne(),
        errors::Internal("Expected shared-ownership of data_by_example."));

    double total_primal_loss = 0;
    double total_dual_loss = 0;
    double total_example_weight = 0;
    OP_REQUIRES_OK(context,
                   data_by_example->Visit([&](const DataByExample::Data& data) {
                     total_primal_loss += data.primal_loss;
                     total_dual_loss += data.dual_loss;
                     total_example_weight += data.example_weight;
                   }));

    // TODO(sibyl-Mooth6ku): Think about the most arithmetically stable way of
    // computing (dual + primal) loss (if it matters).

    {
      Tensor* tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("primal_loss", {}, &tensor));
      tensor->scalar<double>()() = total_primal_loss;
    }

    {
      Tensor* tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("dual_loss", {}, &tensor));
      tensor->scalar<double>()() = total_dual_loss;
    }

    {
      OP_REQUIRES(
          context, total_example_weight > 0,
          errors::FailedPrecondition(
              "No examples found or all examples have zero weight. Either the "
              "optimizer was trained with no instances or perhaps there is a "
              "bug in the training data."));

      Tensor* tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("example_weights", {}, &tensor));
      tensor->scalar<double>()() = total_example_weight;
    }

    // TODO(sibyl-Mooth6ku): Use core::ScopedUnref once it's moved out of internal.
    data_by_example->Unref();
  }

 private:
  string container_;
  string solver_uuid_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaTrainingStats").Device(DEVICE_CPU),
                        SdcaTrainingStats);

}  // namespace tensorflow
