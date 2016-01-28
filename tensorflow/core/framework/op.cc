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

#include "tensorflow/core/framework/op.h"

#include <algorithm>
#include <memory>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// OpRegistry -----------------------------------------------------------------

OpRegistryInterface::~OpRegistryInterface() {}

OpRegistry::OpRegistry() : initialized_(false) {}

void OpRegistry::Register(const OpDef& op_def) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_def)) << "Attempting to register: "
                                                << SummarizeOpDef(op_def);
  } else {
    deferred_.push_back(op_def);
  }
}

const OpDef* OpRegistry::LookUp(const string& op_type_name,
                                Status* status) const {
  const OpDef* op_def = nullptr;
  bool first_call = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = CallDeferred();
    op_def = gtl::FindWithDefault(registry_, op_type_name, nullptr);
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(ValidateKernelRegistrations(this));
  }
  if (op_def == nullptr) {
    status->Update(
        errors::NotFound("Op type not registered '", op_type_name, "'"));
    VLOG(1) << status->ToString();
    static bool first_unregistered = true;
    if (first_unregistered) {
      OpList op_list;
      Export(true, &op_list);
      VLOG(1) << "All registered Ops:";
      for (const auto& op : op_list.op()) {
        VLOG(1) << SummarizeOpDef(op);
      }
      first_unregistered = false;
    }
  }
  return op_def;
}

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs) {
  mutex_lock lock(mu_);
  CallDeferred();
  for (auto p : registry_) {
    op_defs->push_back(*p.second);
  }
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
  mutex_lock lock(mu_);
  CallDeferred();

  std::vector<std::pair<string, const OpDef*>> sorted(registry_.begin(),
                                                      registry_.end());
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
    if (include_internal || !StringPiece(item.first).starts_with("_")) {
      *out->Add() = *item.second;
    }
  }
}

string OpRegistry::DebugString(bool include_internal) const {
  OpList op_list;
  Export(include_internal, &op_list);
  string ret;
  for (const auto& op : op_list.op()) {
    strings::StrAppend(&ret, SummarizeOpDef(op), "\n");
  }
  return ret;
}

bool OpRegistry::CallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (const auto& op_def : deferred_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_def)) << "Attempting to register: "
                                                << SummarizeOpDef(op_def);
  }
  deferred_.clear();
  return true;
}

Status OpRegistry::RegisterAlreadyLocked(const OpDef& def) const {
  TF_RETURN_IF_ERROR(ValidateOpDef(def));

  std::unique_ptr<OpDef> copy(new OpDef(def));
  if (gtl::InsertIfNotPresent(&registry_, def.name(), copy.get())) {
    copy.release();  // Ownership transferred to op_registry
    return Status::OK();
  } else {
    return errors::AlreadyExists("Op with name ", def.name());
  }
}

// static
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}

namespace register_op {
OpDefBuilderReceiver::OpDefBuilderReceiver(const OpDefBuilder& builder) {
  OpDef op_def;
  builder.Finalize(&op_def);
  OpRegistry::Global()->Register(op_def);
}
}  // namespace register_op

extern "C" void RegisterOps(void* registry_ptr) {
  OpRegistry* op_registry = static_cast<OpRegistry*>(registry_ptr);
  std::vector<OpDef> op_defs;
  OpRegistry::Global()->GetRegisteredOps(&op_defs);
  for (auto const& op_def : op_defs) {
    op_registry->Register(op_def);
  }
}

extern "C" void GetOpList(void* str) {
  OpList op_list;
  OpRegistry::Global()->Export(true, &op_list);
  op_list.SerializeToString(reinterpret_cast<string*>(str));
}

}  // namespace tensorflow
