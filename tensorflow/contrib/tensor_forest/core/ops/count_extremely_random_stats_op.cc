// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// CountExtremelyRandomStats outputs count-deltas that should be added to
// the node pcws, candidate split pcws, and total split pcws.  It also outputs
// the leaves that each input arrived to for use in SampleInputs.  This is the
// only op that involves tree traversal, and is constructed so that it can
// be run in parallel on separate batches of data.
#include <unordered_map>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

using std::get;
using std::make_tuple;
using std::pair;
using std::tuple;

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;
using tensorforest::LEAF_NODE;
using tensorforest::FREE_NODE;

using tensorforest::DecideNode;
using tensorforest::Initialize;
using tensorforest::IsAllInitialized;

REGISTER_OP("CountExtremelyRandomStats")
  .Attr("num_classes: int32")
  .Input("input_data: float")

  .Input("input_labels: int32")

  .Input("tree: int32")
  .Input("tree_thresholds: float")

  .Input("node_to_accumulator: int32")

  .Input("candidate_split_features: int32")
  .Input("candidate_split_thresholds: float")

  .Output("pcw_node_delta: float")
  .Output("pcw_splits_indices: int32")
  .Output("pcw_candidate_splits_delta: float")
  .Output("pcw_totals_indices: int32")
  .Output("pcw_total_splits_delta: float")

  .Output("leaves: int32")
  .Doc(R"doc(
   Calculates incremental statistics for a batch of training data.

   Each training example in `input_data` is sent through the decision tree
   represented by `tree` and `tree_thresholds`.  `pcw_node_delta[i]` is
   incremented for every node i that it passes through, and the leaf it ends up
   in is recorded in `leaves[i]`.  Then, if the leaf is fertile and
   initialized, the statistics for its corresponding accumulator slot
   are updated in in `pcw_candidate_splits_delta` and `pcw_total_splits_delta`.

   The attr `num_classes` is needed to appropriately size the outputs.

   input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
   input_labels: The training batch's labels; `input_labels[i]` is the class
     of the i-th input.
   tree:= A 2-d int32 tensor.  `tree[0][i]` gives the index of the left child
     of the i-th node, `tree[0][i] + 1` gives the index of the right child of
     the i-th node, and `tree[1][i]` gives the index of the feature used to
     split the i-th node.
   tree_thresholds: `tree_thresholds[i]` is the value used to split the i-th
     node.
   node_to_accumulator: If the i-th node is fertile, `node_to_accumulator[i]`
     is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
   candidate_split_features: `candidate_split_features[a][s]` is the
     index of the feature being considered by split s of accumulator slot a.
   candidate_split_thresholds: `candidate_split_thresholds[a][s]` is the
     threshold value being considered by split s of accumulator slot a.
   pcw_node_delta: `pcw_node_delta[i][c]` is the number of training examples
     in this training batch with class c that passed through node i.
   pcw_splits_indices:= A 2-d tensor of shape (?, 3).
     `pcw_splits_indices[i]` gives the coordinates of an entry in
     candidate_split_per_class_weights that needs to be updated.
     This is meant to be passed with `pcw_candidate_splits_delta` to a
     scatter_add for candidate_split_per_class_weights:
       training_ops.scatter_add_ndim(candidate_split_per_class_weights,
           pcw_splits_indices, pcw_candidate_splits_delta)
   pcw_candidate_splits_delta: `pcw_candidate_splits_delta[i]` is the
     number of training examples in this training batch that correspond to
     the i-th entry in `pcw_splits_indices` which took the *left* branch of
     candidate split.
   pcw_totals_indices: 'pcw_totals_indices` contains the indices (accumulator,
     class) into total_per_class_weights to update with pcw_total_splits_delta.
   pcw_total_splits_delta: `pcw_total_splits_delta[i]` is the number of
     training examples in this batch that ended up in the fertile
     node with accumulator and class indicated by `pcw_totals_indices[i]`.
   leaves: `leaves[i]` is the leaf that input i ended up in.
)doc");


class CountExtremelyRandomStats : public OpKernel {
 public:
  explicit CountExtremelyRandomStats(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "num_classes", &num_classes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& input_labels = context->input(1);
    const Tensor& tree_tensor = context->input(2);
    const Tensor& tree_thresholds = context->input(3);
    const Tensor& node_to_accumulator = context->input(4);
    const Tensor& candidate_split_features = context->input(5);
    const Tensor& candidate_split_thresholds = context->input(6);

    // Check inputs.
    OP_REQUIRES(context, input_data.shape().dims() == 2,
                errors::InvalidArgument(
                    "input_data should be two-dimensional"));
    OP_REQUIRES(context, input_labels.shape().dims() == 1,
                errors::InvalidArgument(
                    "input_labels should be one-dimensional"));

    OP_REQUIRES(context, tree_tensor.shape().dims() == 2,
            errors::InvalidArgument(
                "tree should be two-dimensional"));
    OP_REQUIRES(context, tree_thresholds.shape().dims() == 1,
            errors::InvalidArgument(
                "tree_thresholds should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
            errors::InvalidArgument(
                "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, candidate_split_features.shape().dims() == 2,
            errors::InvalidArgument(
                "candidate_split_features should be two-dimensional"));
    OP_REQUIRES(context, candidate_split_thresholds.shape().dims() == 2,
            errors::InvalidArgument(
                "candidate_split_thresholds should be two-dimensional"));

    OP_REQUIRES(
        context,
        input_data.shape().dim_size(0) == input_labels.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of inputs should be the same in "
            "input_data and input_labels."));
    OP_REQUIRES(
        context,
        tree_tensor.shape().dim_size(0) ==
        tree_thresholds.shape().dim_size(0) &&
        tree_tensor.shape().dim_size(0) ==
        node_to_accumulator.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of nodes should be the same in "
            "tree, tree_thresholds, and node_to_accumulator"));
    OP_REQUIRES(
        context,
        candidate_split_features.shape() == candidate_split_thresholds.shape(),
        errors::InvalidArgument(
            "candidate_split_features and candidate_split_thresholds should be "
            "the same shape."));

    const int32 num_splits = candidate_split_features.shape().dim_size(1);

    // node pcw delta
    Tensor* output_node_pcw_delta = nullptr;
    TensorShape node_pcw_shape;
    node_pcw_shape.AddDim(tree_tensor.shape().dim_size(0));
    node_pcw_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, node_pcw_shape,
                                            &output_node_pcw_delta));
    Initialize<float>(*output_node_pcw_delta, 0);
    auto out_node = output_node_pcw_delta->tensor<float, 2>();

    // leaves
    Tensor* output_leaves = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(5, input_labels.shape(),
                                            &output_leaves));
    auto out_leaves = output_leaves->unaligned_flat<int32>();

    const auto tree = tree_tensor.tensor<int32, 2>();
    const auto thresholds = tree_thresholds.unaligned_flat<float>();
    const auto labels = input_labels.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto split_features = candidate_split_features.tensor<int32, 2>();
    const auto split_thresholds = candidate_split_thresholds.tensor<float, 2>();

    const int32 num_data = input_data.shape().dim_size(0);

    // <accumulator, class> -> count delta
    std::unordered_map<pair<int32, int32>, int32, PairIntHash> total_delta;
    // <accumulator, split, class> -> count delta
    std::unordered_map<tuple<int32, int32, int32>,
        int32, TupleIntHash> split_delta;
    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i+1);
      int node_index = 0;
      while (true) {
        const int32 label = labels(i);
        ++out_node(node_index, label);
        int32 left_child = tree(node_index, CHILDREN_INDEX);
        if (left_child == LEAF_NODE) {
          out_leaves(i) = node_index;
          const int32 accumulator = node_map(node_index);
          // If the leaf is not fertile or is not yet initialized, we don't
          // count it in the candidate/total split per-class-weights because
          // it won't have any candidate splits yet.
          if (accumulator >= 0 &&
              IsAllInitialized(
                  candidate_split_features.Slice(accumulator,
                                                 accumulator + 1))) {
            ++total_delta[std::make_pair(accumulator, label)];
            for (int split = 0; split < num_splits; split++) {
              if (!DecideNode(point, split_features(accumulator, split),
                              split_thresholds(accumulator, split))) {
                ++split_delta[make_tuple(accumulator, split, label)];
              }
            }
          }
          break;
        } else if (left_child == FREE_NODE) {
          LOG(ERROR) << "Reached a free node, not good.";
          out_leaves(i) = FREE_NODE;
          break;
        }
        node_index = left_child +
            DecideNode(point, tree(node_index, FEATURE_INDEX),
                       thresholds(node_index));
      }
    }

     // candidate splits pcw indices
    Tensor* output_candidate_pcw_indices = nullptr;
    TensorShape candidate_pcw_shape;
    candidate_pcw_shape.AddDim(split_delta.size());
    candidate_pcw_shape.AddDim(3);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, candidate_pcw_shape,
                                            &output_candidate_pcw_indices));
    auto out_candidate_indices =
        output_candidate_pcw_indices->tensor<int32, 2>();

    // candidate splits pcw delta
    Tensor* output_candidate_pcw_delta = nullptr;
    TensorShape candidate_pcw_delta_shape;
    candidate_pcw_delta_shape.AddDim(split_delta.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, candidate_pcw_delta_shape,
                                            &output_candidate_pcw_delta));
    auto out_candidate = output_candidate_pcw_delta->unaligned_flat<float>();

    // total splits indices
    Tensor* output_total_pcw_indices = nullptr;
    TensorShape total_pcw_shape;
    total_pcw_shape.AddDim(total_delta.size());
    total_pcw_shape.AddDim(2);
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, total_pcw_shape,
                                            &output_total_pcw_indices));
    auto out_total_indices = output_total_pcw_indices->tensor<int32, 2>();

    // total splits delta
    Tensor* output_total_pcw_delta = nullptr;
    TensorShape total_pcw_delta_shape;
    total_pcw_delta_shape.AddDim(total_delta.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, total_pcw_delta_shape,
                                            &output_total_pcw_delta));
    auto out_total = output_total_pcw_delta->unaligned_flat<float>();

    // Copy total deltas to output.
    int32 output_slot = 0;
    for (const auto& updates : total_delta) {
      out_total_indices(output_slot, 0) = updates.first.first;
      out_total_indices(output_slot, 1) = updates.first.second;
      out_total(output_slot) = updates.second;
      ++output_slot;
    }

    // Copy split deltas to output.
    output_slot = 0;
    for (const auto& updates : split_delta) {
      out_candidate_indices(output_slot, 0) = get<0>(updates.first);
      out_candidate_indices(output_slot, 1) = get<1>(updates.first);
      out_candidate_indices(output_slot, 2) = get<2>(updates.first);
      out_candidate(output_slot) = updates.second;
      ++output_slot;
    }
  }

 private:
  struct PairIntHash {
   public:
    std::size_t operator()(const std::pair<int, int>& x) const {
      return std::hash<int>()(x.first) ^ std::hash<int>()(x.second);
    }
  };

  struct TupleIntHash {
   public:
    std::size_t operator()(const std::tuple<int32, int32, int32>& x) const {
      return std::hash<int32>()(get<0>(x)) ^ std::hash<int32>()(get<1>(x)) ^
          std::hash<int32>()(get<2>(x));
    }
  };

  int32 num_classes_;
};

REGISTER_KERNEL_BUILDER(Name("CountExtremelyRandomStats").Device(DEVICE_CPU),
                        CountExtremelyRandomStats);

}  // namespace tensorflow
