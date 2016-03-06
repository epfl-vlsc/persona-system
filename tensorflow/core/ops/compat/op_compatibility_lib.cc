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

#include "tensorflow/core/ops/compat/op_compatibility_lib.h"

#include <stdio.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

static string OpsHistoryFile() {
  return strings::StrCat("compat/ops_history.v", TF_MAJOR_VERSION, ".pbtxt");
}

OpCompatibilityLib::OpCompatibilityLib(const string& ops_prefix)
    : ops_file_(io::JoinPath(ops_prefix, "ops.pbtxt")),
      op_history_file_(io::JoinPath(ops_prefix, OpsHistoryFile())) {
  // Get the sorted list of all registered OpDefs.
  printf("Getting all registered ops...\n");
  OpRegistry::Global()->Export(false, &op_list_);
}

Status OpCompatibilityLib::ValidateCompatible(Env* env, int* changed_ops,
                                              int* added_ops,
                                              OpList* out_op_history) {
  *changed_ops = 0;
  *added_ops = 0;

  // Strip docs out of op_list_.
  RemoveDescriptionsFromOpList(&op_list_);

  OpList in_op_history;
  {  // Read op history.
    printf("Reading op history from %s...\n", op_history_file_.c_str());
    string op_history_str;
    Status status = ReadFileToString(env, op_history_file_, &op_history_str);
    if (!errors::IsNotFound(status)) {
      if (!status.ok()) return status;
      protobuf::TextFormat::ParseFromString(op_history_str, &in_op_history);
    }
  }

  int cur = 0;
  int start = 0;
  printf("Verifying updates are compatible...\n");
  // Note: Op history is in (alphabetical, oldest-first) order.
  while (cur < op_list_.op_size() && start < in_op_history.op_size()) {
    if (op_list_.op(cur).name() < in_op_history.op(start).name()) {
      // New op: add it.
      if (out_op_history != nullptr) {
        *out_op_history->add_op() = op_list_.op(cur);
      }
      ++*added_ops;
      ++cur;
    } else if (op_list_.op(cur).name() > in_op_history.op(start).name()) {
      // Op removed: error.
      return errors::InvalidArgument("Error, removed op: ",
                                     SummarizeOpDef(in_op_history.op(start)));
    } else {
      // Op match.

      // Find all historical version of this op.
      const string& op_name = op_list_.op(cur).name();
      int end = start + 1;
      for (; end < in_op_history.op_size(); ++end) {
        if (in_op_history.op(end).name() != op_name) break;
      }

      if (out_op_history != nullptr) {
        // Copy from in_op_history to *out_op_history.
        for (int i = start; i < end; ++i) {
          *out_op_history->add_op() = in_op_history.op(i);
        }
      }

      // Is the last op in the history the same as the current op?
      // Compare using their serialized representations.
      string history_str, cur_str;
      in_op_history.op(end - 1).SerializeToString(&history_str);
      op_list_.op(cur).SerializeToString(&cur_str);

      if (history_str != cur_str) {
        // Op changed, verify the change is compatible.
        for (int i = start; i < end; ++i) {
          TF_RETURN_IF_ERROR(
              OpDefCompatible(in_op_history.op(i), op_list_.op(cur)));
        }

        // Check that attrs missing from in_op_history.op(start) don't
        // change their defaults.
        if (start < end - 1) {
          TF_RETURN_IF_ERROR(OpDefAddedDefaultsUnchanged(
              in_op_history.op(start), in_op_history.op(end - 1),
              op_list_.op(cur)));
        }

        // Compatible! Add changed op to the end of the history.
        if (out_op_history != nullptr) {
          *out_op_history->add_op() = op_list_.op(cur);
        }
        ++*changed_ops;
      }

      // Advance past this op.
      start = end;
      ++cur;
    }
  }

  // Error if missing ops.
  if (start < in_op_history.op_size()) {
    return errors::InvalidArgument("Error, removed op: ",
                                   SummarizeOpDef(in_op_history.op(start)));
  }

  // Add remaining new ops.
  for (; cur < op_list_.op_size(); ++cur) {
    if (out_op_history) {
      *out_op_history->add_op() = op_list_.op(cur);
    }
    ++*added_ops;
  }

  return Status::OK();
}

}  // namespace tensorflow
