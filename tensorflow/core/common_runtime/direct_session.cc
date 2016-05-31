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

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gpu/gpu_tracer.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32 t = options.config.inter_op_parallelism_threads();
  if (t != 0) return t;
  // Default to using the number of cores available in the process.
  return port::NumSchedulableCPUs();
}

thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
  const int32 num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Direct session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(options.env, "Compute", num_threads);
}

thread::ThreadPool* NewThreadPoolFromThreadPoolOptions(
    const SessionOptions& options,
    const ThreadPoolOptionProto& thread_pool_options, int pool_number) {
  int32 num_threads = thread_pool_options.num_threads();
  if (num_threads == 0) {
    num_threads = NumInterOpThreadsFromSessionOptions(options);
  }
  VLOG(1) << "Direct session inter op parallelism threads for pool "
          << pool_number << ": " << num_threads;
  return new thread::ThreadPool(
      options.env, strings::StrCat("Compute", pool_number), num_threads);
}

thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options);
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

std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

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
void DirectSession::SchedClosure(thread::ThreadPool* pool,
                                 std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  pool->Schedule(c);
#endif  // __ANDROID__
}

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : options_(options),
      device_mgr_(device_mgr),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  if (options_.config.session_inter_op_thread_pool_size() > 0) {
    for (int i = 0; i < options_.config.session_inter_op_thread_pool_size();
         ++i) {
      thread_pools_.push_back(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i));
    }
    owns_thread_pools_ = true;
  } else if (options_.config.use_per_session_threads()) {
    thread_pools_.push_back(NewThreadPoolFromSessionOptions(options_));
    owns_thread_pools_ = true;
  } else {
    thread_pools_.push_back(GlobalThreadPool(options));
    owns_thread_pools_ = false;
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
  for (auto& it : partial_runs_) {
    delete it.second;
  }
  for (auto it : executors_) {
    delete it.second;
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  delete cancellation_manager_;
  if (owns_thread_pools_) {
    for (auto* p : thread_pools_) delete p;
  }

  execution_state_.reset(nullptr);
  flib_def_.reset(nullptr);
}

void DirectSession::MaybeInitializeExecutionState(const GraphDef& graph) {
  // If already initialied, do nothing.
  if (flib_def_ && execution_state_) {
    return;
  }
  // Set up the per-session execution state.
  flib_def_.reset(new FunctionLibraryDefinition(graph.library()));
  SimpleGraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &options_;
  execution_state_.reset(
      new SimpleGraphExecutionState(flib_def_.get(), options));
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
  MaybeInitializeExecutionState(graph);
  std::unique_ptr<SimpleGraphExecutionState> old_state;
  SimpleGraphExecutionState* new_state = nullptr;
  TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &new_state));

  // Swap out the old state.
  old_state = std::move(execution_state_);
  execution_state_.reset(new_state);

  graph_created_ = true;  // In case this is first call
  return Status::OK();
}

// TODO(yuanbyu): Simplify by treating Run() as "PRunSetup(); PRun()".
Status DirectSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status DirectSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
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

  if (run_options.inter_op_thread_pool() < 0 ||
      run_options.inter_op_thread_pool() >= thread_pools_.size()) {
    return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                                   run_options.inter_op_thread_pool());
  }
  thread::ThreadPool* pool = thread_pools_[run_options.inter_op_thread_pool()];

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  TF_RETURN_IF_ERROR(
      GetOrCreateExecutors(pool, input_tensor_names, output_names, target_nodes,
                           &executors_and_keys, &run_state_args));

  // Create a run state and start execution.
  RunState run_state(input_tensor_names, output_names);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());

  // Send inputs.
  TF_RETURN_IF_ERROR(SendInputs(inputs, executors_and_keys, run_state.rendez));

  // Start parallel Executors.
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  args.rendezvous = run_state.rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, c);
  };
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);
  const int64 build_cost_model =
      options_.config.graph_options().build_cost_model();
  if (do_trace || build_cost_model > 0) {
    args.stats_collector = new StepStatsCollector(
        run_metadata->mutable_step_stats(),
        (build_cost_model > 0) ? &cost_model_manager_ : nullptr);
    run_state.collector = args.stats_collector;
  }

  // TODO(pbar) CostModel still gets very confused when presented
  // with trace data from the GPUTracer. This will need fixing if the
  // cost model needs meaningful GPU timing information.
  std::unique_ptr<GPUTracer> tracer;
  if (!build_cost_model &&
      run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
    tracer.reset(CreateGPUTracer());
    // tracer will be NULL on non-GPU platforms.
    if (tracer) tracer->Start();
  }

  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  WaitForNotification(&run_state, run_options.timeout_in_ms() > 0
                                      ? run_options.timeout_in_ms()
                                      : operation_timeout_in_ms_);

  if (tracer) {
    tracer->Stop();
    tracer->Collect(args.stats_collector);
  }

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Receive outputs.
  TF_RETURN_IF_ERROR(
      RecvOutputs(output_names, executors_and_keys, &run_state, outputs));

  // Save the output tensors of this run we choose to keep.
  TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(output_names, &session_state_));

  // Build and return the cost model as instructed.
  mutex_lock l(executor_lock_);
  ++executors_and_keys->step_count;
  if (executors_and_keys->step_count == build_cost_model) {
    CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
    for (const auto& item : executors_and_keys->items) {
      TF_RETURN_IF_ERROR(
          cost_model_manager_.AddToCostGraphDef(item.graph, cost_graph));
    }
  }
  return Status::OK();
}

Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle) {
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before PRunSetup()!");
    }
  }

  // RunOptions is not available in PRunSetup, so use thread pool 0.
  thread::ThreadPool* pool = thread_pools_[0];

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  run_state_args.is_partial_run = true;
  Status s = GetOrCreateExecutors(pool, input_names, output_names, target_nodes,
                                  &executors_and_keys, &run_state_args);
  TF_RETURN_IF_ERROR(s);

  // Create the run state and save it for future PRun calls.
  RunState* run_state = new RunState(input_names, output_names);
  run_state->rendez = new IntraProcessRendezvous(device_mgr_.get());
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_.insert({run_state_args.handle, run_state}).second) {
      return errors::Internal("The handle '", run_state_args.handle,
                              "' created for this partial run is not unique.");
    }
  }

  // Start parallel Executors.
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state->rendez, [run_state](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(run_state->mu_);
          run_state->status.Update(ret);
        }
        run_state->executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  args.rendezvous = run_state->rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, c);
  };
  args.session_state = &session_state_;
  args.tensor_store = &run_state->tensor_store;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  if (options_.config.graph_options().build_cost_model()) {
    args.stats_collector =
        new StepStatsCollector(nullptr, &cost_model_manager_);
    run_state->collector = args.stats_collector;
  }

  for (auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return Status::OK();
}

Status DirectSession::PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  ExecutorsAndKeys* executors_and_keys;
  RunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second;

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second;

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " had already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument("The fetch ", output,
                                       " had already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  Status s = SendInputs(inputs, executors_and_keys, run_state->rendez);

  // Receive outputs.
  if (s.ok()) {
    s = RecvOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Save the output tensors of this run we choose to keep.
  if (s.ok()) {
    s = run_state->tensor_store.SaveTensors(output_names, &session_state_);
  }

  {
    mutex_lock l(executor_lock_);
    // Delete the run state if there is an error or all fetches are done.
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu_);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& it : inputs) {
        run_state->pending_inputs.erase(it.first);
      }
      for (const auto& name : output_names) {
        run_state->pending_outputs.erase(name);
      }
      done = (run_state->pending_inputs.size() == 0 &&
              run_state->pending_outputs.size() == 0);
    }
    if (done) {
      WaitForNotification(run_state, operation_timeout_in_ms_);
      partial_runs_.erase(handle);
      delete run_state;
    }
  }
  return s;
}

Status DirectSession::SendInputs(const NamedTensorList& inputs,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 IntraProcessRendezvous* rendez) {
  Status s;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it = executors_and_keys->input_keys.find(input.first);
    if (it == executors_and_keys->input_keys.end()) {
      return errors::InvalidArgument("'", input.first,
                                     "' is not a pre-defined feed!");
    }
    const string& input_key = it->second;
    s = rendez->Send(input_key, Rendezvous::Args(), input.second, false);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return Status::OK();
}

Status DirectSession::RecvOutputs(const std::vector<string>& output_names,
                                  const ExecutorsAndKeys* executors_and_keys,
                                  RunState* run_state,
                                  std::vector<Tensor>* outputs) {
  Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it = executors_and_keys->output_keys.find(output_name);
    if (it == executors_and_keys->output_keys.end()) {
      return errors::InvalidArgument("'", output_name,
                                     "' was not defined as a fetch"
                                     " target in PRunSetup.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;

    // Fetch data from the Rendezvous.
    IntraProcessRendezvous* rendez = run_state->rendez;
    s = rendez->Recv(output_key, Rendezvous::Args(), &output_tensor, &is_dead);
    if (is_dead && s.ok()) {
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
  return Status::OK();
}

Status DirectSession::CheckFetch(const NamedTensorList& feeds,
                                 const std::vector<string>& fetches,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 const RunState* run_state) {
  const Graph* graph = executors_and_keys->graph;
  const NameNodeMap* name_to_node = executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const string& feed : run_state->pending_inputs) {
      TensorId id(ParseTensorName(feed));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", feed, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return Status::OK();
}

Status DirectSession::GetOrCreateExecutors(
    thread::ThreadPool* pool, gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<string> outputs, gtl::ArraySlice<string> target_nodes,
    ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args) {
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
                                     str_util::Join(tn_sorted, ","), "/",
                                     run_state_args->is_partial_run);

  // Set the handle.
  {
    mutex_lock l(mu_);
    run_state_args->handle = strings::StrCat(key, ";", name_counter_++);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second;
      return Status::OK();
    }
  }

  BuildGraphOptions options;
  options.feed_endpoints = inputs_sorted;
  options.fetch_endpoints = outputs_sorted;
  options.target_nodes = tn_sorted;

  // The executor_lock_ is intentionally released while executor is
  // being created.
  std::unordered_map<string, Graph*> graphs;
  Status s = CreateGraphs(options, &graphs, run_state_args);
  TF_RETURN_IF_ERROR(s);

  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
  ek->func_defs = flib_def_.get();
  if (run_state_args->is_partial_run) {
    ek->graph = run_state_args->graph;
    ek->name_to_node = new NameNodeMap;
    std::unordered_set<StringPiece, StringPiece::Hasher> names;
    for (const string& input : inputs) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : outputs) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : run_state_args->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node->insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());
  auto runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, c);
  };
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    Graph* partition_graph = iter->second;
    const int graph_def_version = partition_graph->versions().producer();

    Device* device;
    s = device_mgr_->LookupDevice(partition_name, &device);
    if (!s.ok()) break;

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    item->flib = NewFunctionLibraryRuntime(device_mgr_.get(), device, runner,
                                           graph_def_version, flib_def_.get(),
                                           optimizer_opts);

    LocalExecutorParams params;
    params.device = device;
    params.function_library = item->flib;
    auto lib = item->flib;
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // Caches the kernel only if the node is stateful.
      if (!lib->IsStateful(ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };

    optimizer.Optimize(lib, device, &partition_graph);
    s = EnsureMemoryTypes(DeviceType(device->device_type()), device->name(),
                          partition_graph);
    if (!s.ok()) {
      break;
    }
    // NewLocalExecutor takes ownership of *partition_graph.
    iter->second = nullptr;
    item->graph = partition_graph;
    item->executor = nullptr;
    s = NewLocalExecutor(params, partition_graph, &item->executor);
    if (!s.ok()) {
      break;
    }
  }
  if (!s.ok()) {
    gtl::STLDeleteValues(&graphs);
    return s;
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

Status DirectSession::CreateGraphs(const BuildGraphOptions& options,
                                   std::unordered_map<string, Graph*>* outputs,
                                   RunStateArgs* run_state_args) {
  std::unique_ptr<SimpleClientGraph> client_graph;
  SimpleClientGraph* cgraph = nullptr;

  std::unique_ptr<SimpleGraphExecutionState> temp_exec_state_holder;
  SimpleGraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new SimpleGraphExecutorState for every new unseen graph,
    // and then place it.
    SimpleGraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    temp_exec_state_holder.reset(
        new SimpleGraphExecutionState(flib_def_.get(), prune_options));
    {
      mutex_lock l(mu_);
      temp_exec_state_holder->SetStatefulPlacements(stateful_placements_);
    }

    TF_RETURN_IF_ERROR(temp_exec_state_holder->Extend(
        execution_state_->original_graph_def(), &execution_state));
    temp_exec_state_holder.reset(execution_state);
  } else {
    execution_state = execution_state_.get();
  }

  TF_RETURN_IF_ERROR(execution_state->BuildGraph(options, &cgraph));
  {
    auto current_stateful_placements = execution_state->GetStatefulPlacements();
    mutex_lock l(mu_);
    // Update our current state based on the execution_state's
    // placements.  If there are any mismatches for a node,
    // we should fail, as this should never happen.
    for (auto placement_pair : current_stateful_placements) {
      const string& node_name = placement_pair.first;
      const string& placement = placement_pair.second;
      auto iter = stateful_placements_.find(node_name);
      if (iter == stateful_placements_.end()) {
        stateful_placements_.insert(std::make_pair(node_name, placement));
      } else if (iter->second != placement) {
        return errors::Internal(
            "Stateful placement mismatch. "
            "Current assignment of ",
            node_name, " to ", iter->second, " does not match ", placement);
      }
    }

    stateful_placements_ = execution_state->GetStatefulPlacements();
  }
  client_graph.reset(cgraph);

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph = new Graph(flib_def_.get());
    CopyGraph(*execution_state->full_graph(), run_state_args->graph);
  }

  // Partition the graph across devices.
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

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

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

  Status s;
  for (auto&& partition : partitions) {
    const string& partition_name = partition.first;

    GraphDef* graph_def = &partition.second;
    VLOG(2) << "Created " << ProtoDebugString(*graph_def) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    {
      mutex_lock l(graph_def_lock_);
      // TODO(pbar) The library is currently shared and immutable. There
      // may be possible use cases where a device may want to modify
      // function definitions - in which case the library would need to be
      // replicated per device.
      s = d->MaybeRewriteGraph(flib_def_->ToProto(), graph_def);
      if (!s.ok()) {
        break;
      }
    }
    Graph* device_graph = new Graph(flib_def_.get());
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    s = ConvertGraphDefToGraph(device_opts, *graph_def, device_graph);
    if (!s.ok()) {
      delete device_graph;
      break;
    }
    outputs->insert(std::make_pair(partition_name, device_graph));
  }
  if (!s.ok()) {
    // Also delete other graphs created during the loop.
    gtl::STLDeleteValues(outputs);
    return s;
  }
  return Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  return ::tensorflow::Status::OK();
}

DirectSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
  if (collector != nullptr) {
    delete collector;
  }
}

void DirectSession::WaitForNotification(RunState* run_state,
                                        int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    bool timed_out =
        run_state->executors_done.WaitForNotificationWithTimeout(timeout_in_ms);
    if (timed_out) {
      {
        mutex_lock l(run_state->mu_);
        run_state->status.Update(Status(error::DEADLINE_EXCEEDED,
                                        "Timed out waiting for notification"));
      }
      // TODO(sherrym): This cancels all steps in the session, even ones that
      // have not exceeded their deadline. An alternative would be to use a
      // two-level cancellation manager with a Session-global one containing
      // several step-local ones. Probably the RunState should have its own
      // CancellationManager.
      cancellation_manager_->StartCancel();
    }
  } else {
    run_state->executors_done.WaitForNotification();
  }
}

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty();
  }

  Session* NewSession(const SessionOptions& options) override {
    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }
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
