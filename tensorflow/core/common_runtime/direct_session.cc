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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

thread::ThreadPool* NewThreadPool(const SessionOptions& options) {
  int32 inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    // Default to using the number of cores available in the process.
    inter_op_parallelism_threads = port::NumSchedulableCPUs();
  }
  VLOG(1) << "Direct session inter op parallelism threads: "
          << inter_op_parallelism_threads;
  return new thread::ThreadPool(options.env, "Compute",
                                inter_op_parallelism_threads);
}

thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  static thread::ThreadPool* const thread_pool = NewThreadPool(options);
  return thread_pool;
}

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void DirectSession::SchedClosure(std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  thread_pool_->Schedule(c);
#endif  // __ANDROID__
}

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : options_(options),
      device_mgr_(device_mgr),
      cancellation_manager_(new CancellationManager()) {
  if (options_.config.use_per_session_threads()) {
    thread_pool_ = NewThreadPool(options_);
  } else {
    thread_pool_ = GlobalThreadPool(options);
  }
  // NOTE(mrry): We do not need to use a unique string for the session
  // handle, because DirectSession owns its devices. This may change
  // in future versions.
  session_handle_ = "direct";
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

DirectSession::~DirectSession() {
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  for (auto it : executors_) {
    delete it.second;
  }
  delete cancellation_manager_;

  if (options_.config.use_per_session_threads()) {
    delete thread_pool_;
  }
}

Status DirectSession::Create(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  if (graph_created_) {
    return errors::AlreadyExists(
        "A Graph has already been created for this session.");
  }
  return ExtendLocked(graph);
}

Status DirectSession::Extend(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

Status DirectSession::ExtendLocked(const GraphDef& graph) {
  if (graph_created_ && graph_def_.version() != graph.version()) {
    return errors::InvalidArgument("Incompatible GraphDef versions in Extend: ",
                                   graph_def_.version(), " != ",
                                   graph.version());
  }

  const int node_size_before_merge = graph_def_.node_size();
  graph_def_.MergeFrom(graph);

  FunctionLibraryDefinition fdefs(graph_def_.library());
  // Add default attributes to all new nodes in the graph.
  Status s =
      AddDefaultAttrsToGraphDef(&graph_def_, &fdefs, node_size_before_merge);
  if (!s.ok()) {
    // One of the nodes was invalid, return the state of graph_def_
    // to what it was before this function.
    const int nodes_added = graph_def_.node_size() - node_size_before_merge;
    graph_def_.mutable_node()->DeleteSubrange(node_size_before_merge,
                                              nodes_added);
    return s;
  }

  if (graph_def_.version() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(graph_def_, &fdefs));
  }

  graph_created_ = true;  // In case this is first call
  return Status::OK();
}

Status DirectSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before Run()!");
    }
  }

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  Status s = GetOrCreateExecutors(input_tensor_names, output_names,
                                  target_nodes, &executors_and_keys);
  if (!s.ok()) {
    return s;
  }

  IntraProcessRendezvous* rendez =
      new IntraProcessRendezvous(device_mgr_.get());
  core::ScopedUnref rendez_unref(rendez);

  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    const string& input_key = executors_and_keys->input_keys[input.first];
    s = rendez->Send(input_key, Rendezvous::Args(), input.second, false);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }

  // Start parallel Executors.
  Notification executors_done;
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, rendez, [&executors_done, &s](const Status& ret) {
        s = ret;
        executors_done.Notify();
      });

  Executor::Args args;
  args.rendezvous = rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this](Executor::Args::Closure c) { SchedClosure(c); };

  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  executors_done.WaitForNotification();

  TF_RETURN_IF_ERROR(s);

  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_key =
        executors_and_keys->output_keys[output_names[output_offset]];
    Tensor output_tensor;
    bool is_dead;

    // Fetch data from the Rendezvous.
    s = rendez->Recv(output_key, Rendezvous::Args(), &output_tensor, &is_dead);
    if (is_dead) {
      s = errors::InvalidArgument("The tensor returned for ",
                                  output_names[output_offset],
                                  " was not valid.");
    }
    if (!s.ok()) {
      rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }

  return s;
}

Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes,
    ExecutorsAndKeys** executors_and_keys) {
  // Sort the inputs and outputs, so we don't create separate
  // executors when a user passes in the same inputs/outputs in
  // different orders.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string key = strings::StrCat(str_util::Join(inputs_sorted, ","), "->",
                                     str_util::Join(outputs_sorted, ","), "/",
                                     str_util::Join(tn_sorted, ","));

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second;
      return Status::OK();
    }
  }

  // The executor_lock_ is intentionally released while executor is
  // being created.
  FunctionLibraryDefinition* fdefs;
  std::unordered_map<string, Graph*> graphs;
  Status s = CreateGraphs(inputs, outputs, target_nodes, &fdefs, &graphs);
  if (!s.ok()) {
    return s;
  }

  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
  ek->func_defs = fdefs;
  ek->items.reserve(graphs.size());
  auto runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  for (const auto& graph : graphs) {
    const string& partition_name = graph.first;
    Graph* partition_graph = graph.second;
    const int graph_def_version = partition_graph->version();

    Device* device;
    s = device_mgr_->LookupDevice(partition_name, &device);
    if (!s.ok()) {
      return s;
    }

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    item->flib =
        NewFunctionLibraryRuntime(device, runner, graph_def_version, fdefs);

    LocalExecutorParams params;
    params.device = device;
    params.function_library = item->flib;
    auto lib = item->flib;
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [](OpKernel* kernel) {
      // Do nothing because 'kernel' is owned by opseg above.
    };

    s = NewLocalExecutor(params, partition_graph, &item->executor);
    if (!s.ok()) {
      return s;
    }
  }

  // Compute the rendezvous keys to avoid recomputing them every time.
  //
  // We always use the first device as the device name portion of the
  // key, even if we're feeding another graph.
  for (const string& input : inputs) {
    ek->input_keys[input] = GetRendezvousKey(
        input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }
  for (const string& output : outputs) {
    ek->output_keys[output] = GetRendezvousKey(
        output, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  const bool inserted = executors_.insert(std::make_pair(key, ek.get())).second;
  if (!inserted) {
    // Another thread created the entry before us, so delete the
    // one we created and return the already created one.
    auto it = executors_.find(key);
    *executors_and_keys = it->second;
  } else {
    *executors_and_keys = ek.release();
  }

  return Status::OK();
}

void DirectSession::SaveStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      VLOG(2) << "Saving " << n->DebugString();
      stateful_placements_[n->name()] = n->assigned_device_name();
    }
  }
}

void DirectSession::RestoreStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      auto iter = stateful_placements_.find(n->name());
      if (iter != stateful_placements_.end()) {
        n->set_assigned_device_name(iter->second);
        VLOG(2) << "Restored " << n->DebugString();
      }
    }
  }
}

Status DirectSession::CreateGraphs(
    gtl::ArraySlice<string> feeds, gtl::ArraySlice<string> fetches,
    gtl::ArraySlice<string> target_nodes, FunctionLibraryDefinition** func_defs,
    std::unordered_map<string, Graph*>* outputs) {
  std::unique_ptr<FunctionLibraryDefinition> fdefs;
  std::unique_ptr<Graph> graph;
  GraphConstructorOptions opts;
  if (options_.config.has_graph_options()) {
    opts.optimizer_do_cse = !options_.config.graph_options()
                                 .skip_common_subexpression_elimination();
  } else {
    opts.optimizer_do_cse = true;
  }

  if (opts.optimizer_do_cse) {
    // Prevent CSE from eliminating nodes that will be required during
    // RewriteGraphForExecution, below.
    std::unordered_set<StringPiece, StringPiece::Hasher> no_cse_nodes;
    for (const string& feed : feeds) {
      no_cse_nodes.insert(ParseTensorName(feed).first);
    }
    for (const string& fetch : fetches) {
      no_cse_nodes.insert(ParseTensorName(fetch).first);
    }
    for (const string& target_node : target_nodes) {
      no_cse_nodes.insert(target_node);
    }
    opts.cse_consider_function = [no_cse_nodes](const Node* n) {
      return n->type_string() != "Const" && !no_cse_nodes.count(n->name());
    };
  }

  {
    mutex_lock l(graph_def_lock_);
    fdefs.reset(new FunctionLibraryDefinition(graph_def_.library()));
    graph.reset(new Graph(fdefs.get()));
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def_, graph.get()));
  }

  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph.get(), feeds, fetches, target_nodes,
      device_set_.client_device()->attributes()));

  // Run the simple placer after rewriting the graph.
  std::unordered_map<string, int32> node_name_to_cost_map;
  for (Node* n : graph->nodes()) {
    node_name_to_cost_map[n->name()] = n->cost_id();
  }
  SimplePlacer placer(graph.get(), &device_set_, &node_name_to_cost_map,
                      &options_);

  {
    mutex_lock l(mu_);
    // Restore stateful nodes.
    RestoreStatefulNodes(graph.get());
    TF_RETURN_IF_ERROR(placer.Run());
    // Save stateful nodes.
    SaveStatefulNodes(graph.get());
  }

  // Partition the graph across devices.
  std::unordered_map<string, GraphDef> partitions;
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    mutex_lock l(mu_);
    return strings::StrCat(prefix, "/_", name_counter_++);
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.control_flow_added = false;
  TF_RETURN_IF_ERROR(Partition(popts, graph.get(), &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string& local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  for (auto partition : partitions) {
    const string& partition_name = partition.first;

    GraphDef* graph_def = &partition.second;
    VLOG(2) << "Created " << graph_def->DebugString() << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &d));
    {
      mutex_lock l(graph_def_lock_);
      // TODO(pbar) The library is currently shared and immutable. There
      // may be possible use cases where a device may want to modify
      // function definitions - in which case the library would need to be
      // replicated per device.
      TF_RETURN_IF_ERROR(d->MaybeRewriteGraph(graph_def_.library(), graph_def));
    }
    Graph* device_graph = new Graph(fdefs.get());
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    Status s = ConvertGraphDefToGraph(device_opts, *graph_def, device_graph);
    if (!s.ok()) {
      delete device_graph;
      // Also delete other graphs created during the loop.
      gtl::STLDeleteValues(outputs);
      return s;
    }
    outputs->insert(std::make_pair(partition_name, device_graph));
  }
  *func_defs = fdefs.release();
  return Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  return ::tensorflow::Status::OK();
}

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  Session* NewSession(const SessionOptions& options) override {
    std::vector<Device*> devices;
    DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0",
                              &devices);
    return new DirectSession(options, new DeviceMgr(devices));
  }
};

class DirectSessionRegistrar {
 public:
  DirectSessionRegistrar() {
    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
  }
};
static DirectSessionRegistrar registrar;

}  // namespace tensorflow
