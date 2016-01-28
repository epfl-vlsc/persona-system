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

#include "tensorflow/core/framework/graph_def_util.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

string SummarizeGraphDef(const GraphDef& graph_def) {
  string ret;
  strings::StrAppend(&ret, "versions = ",
                     graph_def.versions().ShortDebugString(), ";\n");
  for (const NodeDef& node : graph_def.node()) {
    strings::StrAppend(&ret, SummarizeNodeDef(node), ";\n");
  }
  return ret;
}

Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def) {
  for (const NodeDef& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateExternalNodeDefSyntax(node));
  }
  return Status::OK();
}

Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                 const OpRegistryInterface* op_registry,
                                 int node_offset) {
  if (node_offset > graph_def->node_size()) {
    return errors::InvalidArgument(
        "Tried to add default attrs to GraphDef "
        "starting at offset ",
        node_offset, " with total nodes in graph: ", graph_def->node_size());
  }

  Status s;
  for (int i = node_offset; i < graph_def->node_size(); ++i) {
    NodeDef* node_def = graph_def->mutable_node(i);
    const OpDef* op_def = op_registry->LookUp(node_def->op(), &s);
    if (!s.ok()) {
      return s;
    }
    AddDefaultsToNodeDef(*op_def, node_def);
  }

  return s;
}

}  // namespace tensorflow
