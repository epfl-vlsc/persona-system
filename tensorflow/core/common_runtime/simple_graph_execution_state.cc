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

#include "tensorflow/core/common_runtime/simple_graph_execution_state.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

SimpleGraphExecutionState::SimpleGraphExecutionState(
    const OpRegistryInterface* ops,
    const SimpleGraphExecutionStateOptions& options)
    : ops_(ops),
      device_set_(options.device_set),
      session_options_(options.session_options),
      graph_(nullptr) {
  // TODO(mrry): Publish placement visualizations or handle the log
  // placement option.
}

SimpleGraphExecutionState::~SimpleGraphExecutionState() {
  mutex_lock l(mu_);
  delete graph_;
}

Status SimpleGraphExecutionState::Create(GraphDef* graph_def) {
  if (original_graph_def_.node_size() > 0) {
    return errors::InvalidArgument(
        "Cannot call Create on SimpleGraphExecutionState twice");
  }

  original_graph_def_.Swap(graph_def);
  VLOG(2) << "Incoming def: " << ProtoDebugString(original_graph_def_);
  return AddDefaultAttrsToGraphDef(&original_graph_def_, *ops_, 0);
}

Status SimpleGraphExecutionState::Extend(
    const GraphDef& extension_def, SimpleGraphExecutionState** out) const {
  std::unordered_set<string> new_names;
  // 1. Build an index of the new node names.
  for (const NodeDef& node : extension_def.node()) {
    new_names.insert(node.name());
  }

  // 2. Add the non-duplicates from the old graph to the new graph.
  //    Return an error if the same node name appears in both the
  //    old graph and the extension.
  GraphDef gdef;
  for (const NodeDef& node : original_graph_def_.node()) {
    if (new_names.count(node.name()) == 0) {
      *gdef.add_node() = node;
    } else {
      return errors::InvalidArgument(tensorflow::strings::Printf(
          "GraphDef argument to Extend includes node '%s', which was created "
          "by a previous call to Create or Extend in this session.",
          node.name().c_str()));
    }
  }

  int old_node_size = gdef.node_size();
  gdef.mutable_node()->MergeFrom(extension_def.node());
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&gdef, *ops_, old_node_size));

  // 3. Add the extension.
  SimpleGraphExecutionStateOptions combined_options;
  combined_options.device_set = device_set_;

  SimpleGraphExecutionState* new_execution_state =
      new SimpleGraphExecutionState(ops_, combined_options);
  Status new_execution_state_status = new_execution_state->Create(&gdef);
  if (!new_execution_state_status.ok()) {
    delete new_execution_state;
    return new_execution_state_status;
  }
  *out = new_execution_state;

  // TODO(mrry): This is likely to be used for non-throughput-sensitive
  // interactive workloads, but in future we may want to transfer other
  // parts of the placement and/or cost model.
  return Status::OK();
}

Status SimpleGraphExecutionState::InitBaseGraph() {
  std::unique_ptr<Graph> new_graph(new Graph(ops_));
  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, original_graph_def_, new_graph.get()));
  SimplePlacer placer(new_graph.get(), device_set_, session_options_);
  // TODO(mrry): Consider making the SimplePlacer cancelable.
  TF_RETURN_IF_ERROR(placer.Run());
  graph_ = new_graph.release();
  return Status::OK();
}

Status SimpleGraphExecutionState::BuildGraph(const BuildGraphOptions& options,
                                             SimpleClientGraph** out) {
  VLOG(1) << "BuildGraph";
  mutex_lock l(mu_);
  // Lazily initialize the base graph.
  if (!graph_) {
    TF_RETURN_IF_ERROR(InitBaseGraph());
  }

  std::unique_ptr<SimpleClientGraph> cgraph(new SimpleClientGraph(ops_));
  CopyGraph(*graph_, &cgraph->graph);

  // Extract the subset of the graph that needs to be run, adding feed/fetch
  // ops as needed.
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      &cgraph->graph, options.feed_endpoints, options.fetch_endpoints,
      options.target_nodes, device_set_->client_device()->attributes()));

  // Copy the extracted graph in order to make its node ids dense,
  // since the local CostModel used to record its stats is sized by
  // the largest node id.
  {
    std::unique_ptr<SimpleClientGraph> dense_copy(new SimpleClientGraph(ops_));
    CopyGraph(cgraph->graph, &dense_copy->graph);
    cgraph = std::move(dense_copy);
  }

  // TODO(vrv): We should check invariants of the graph here.

  *out = cgraph.release();

  return Status::OK();
}

}  // namespace tensorflow
