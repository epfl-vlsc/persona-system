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

#ifndef TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_
#define TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class Device;
class ThreadPool;

class DirectSession : public Session {
 public:
  // Takes ownership of 'device_mgr'.
  DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr);
  ~DirectSession() override;

  ::tensorflow::Status Create(const GraphDef& graph) override;
  ::tensorflow::Status Extend(const GraphDef& graph) override;
  ::tensorflow::Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs) override;
  ::tensorflow::Status Close() override;

 private:
  typedef DirectSession ME;

  // We create one executor and its dependent library runtime for
  // every partition.
  struct PerPartitionExecutorsAndLib {
    Executor* executor = nullptr;
    FunctionLibraryRuntime* flib = nullptr;
  };

  // An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 'func_defs' are the function definition used by all the
  // underlying executors. Each item in 'items' is the executor for a
  // partition of the graph bundled with its dependent library
  // runtime. 'input_keys' are the rendezvous keys for the feeds and
  // 'output_keys' are rendezvous keys for the fetches.
  struct ExecutorsAndKeys {
    FunctionLibraryDefinition* func_defs = nullptr;
    std::vector<PerPartitionExecutorsAndLib> items;
    std::unordered_map<string, string> input_keys;
    std::unordered_map<string, string> output_keys;

    ~ExecutorsAndKeys() {
      for (auto item : items) {
        delete item.executor;
        delete item.flib;
      }
      delete func_defs;
    }
  };

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys);

  // Creates several graphs given the existing graph_def_ and the
  // input feeds and fetches, given 'devices'.
  ::tensorflow::Status CreateGraphs(
      gtl::ArraySlice<string> feeds, gtl::ArraySlice<string> fetches,
      gtl::ArraySlice<string> target_nodes,
      FunctionLibraryDefinition** func_defs,
      std::unordered_map<string, Graph*>* outputs);

  ::tensorflow::Status ExtendLocked(const GraphDef& graph)
      EXCLUSIVE_LOCKS_REQUIRED(graph_def_lock_);

  const SessionOptions options_;

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  string session_handle_;
  bool graph_created_ GUARDED_BY(graph_def_lock_) = false;

  mutex graph_def_lock_;
  GraphDef graph_def_ GUARDED_BY(graph_def_lock_);

  // The thread-pool to use for running ops.
  thread::ThreadPool* thread_pool_ = nullptr;

  // Schedules 'c' for execution.
  void SchedClosure(std::function<void()> c);

  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  std::unordered_map<string, ExecutorsAndKeys*> executors_
      GUARDED_BY(executor_lock_);

  CancellationManager* cancellation_manager_;

  // Saves and restores device placements for stateful nodes.
  mutex mu_;
  void SaveStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void RestoreStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_ GUARDED_BY(mu_);

  // For generating unique names.
  int64 name_counter_ GUARDED_BY(mu_) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_
