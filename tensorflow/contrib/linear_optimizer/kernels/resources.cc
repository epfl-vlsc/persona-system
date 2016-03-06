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

#include "tensorflow/contrib/linear_optimizer/kernels/resources.h"

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

DataByExample::DataByExample(const string& container, const string& solver_uuid)
    : container_(container), solver_uuid_(solver_uuid) {}

DataByExample::~DataByExample() {}

// static
DataByExample::Key DataByExample::MakeKey(const string& example_id) {
  // Platform agnostic and collision resistant key-generation function.
  // The current probability of at least one collision for 1B example_ids is
  // approximately 10^-11 (ie 2^60 / 2^97).
  //
  // TODO(katsiapis): Investigate using a smaller key space to save memory if
  // collisions are not much of an issue.
  //
  // TODO(katsiapis): Avoid the double-pass over the data.
  static const uint64 kSeed1 = 0xDECAFCAFFE;  // Same as Hash64 default seed.
  static const uint64 kSeed2 = 0xABCDEF0123;  // Anything != kSeed1 will do.
  return std::make_pair(
      Hash64(example_id.data(), example_id.size(), kSeed1),
      Hash64(example_id.data(), example_id.size(), kSeed2) & 0xFFFFFFFF);
}

DataByExample::Data& DataByExample::operator[](const Key& key) {
  mutex_lock l(mu_);
  return duals_by_key[key];
}

string DataByExample::DebugString() {
  return strings::StrCat("DataByExample(", container_, ", ", solver_uuid_, ")");
}

size_t DataByExample::KeyHash::operator()(const Key& key) const {
  // Since key.first is already a Hash64 it suffices.
  return key.first;
}

}  // namespace tensorflow
