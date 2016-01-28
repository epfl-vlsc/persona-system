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

#include "tensorflow/core/framework/partial_tensor_shape.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

bool PartialTensorShape::IsValid(const TensorShapeProto& proto) {
  for (const auto& d : proto.dim()) {
    if (d.size() < -1) return false;
  }
  return true;
}

bool PartialTensorShape::IsFullyDefined() const {
  for (auto s : dim_sizes_) {
    if (s < 0) return false;
  }
  return true;
}

Status PartialTensorShape::IsValidShape(const TensorShapeProto& proto) {
  for (const auto& d : proto.dim()) {
    if (d.size() < -1) {
      return errors::InvalidArgument(
          "Shape ", DebugString(proto),
          " has dimensions with values below -1 (where -1 means unknown)");
    }
  }
  return Status::OK();
}

PartialTensorShape::PartialTensorShape(const TensorShapeProto& proto) {
  dim_sizes_.reserve(proto.dim_size());
  for (const auto& d : proto.dim()) {
    CHECK_GE(d.size(), -1);
    dim_sizes_.push_back(d.size());
  }
}

PartialTensorShape::PartialTensorShape(gtl::ArraySlice<int64> dim_sizes) {
  dim_sizes_.reserve(dim_sizes.size());
  for (auto s : dim_sizes) {
    CHECK_GE(s, -1);
    dim_sizes_.push_back(s);
  }
}

PartialTensorShape::PartialTensorShape() {}

PartialTensorShape PartialTensorShape::Concatenate(int64 size) const {
  CHECK_GE(size, -1);
  PartialTensorShape out = *this;
  out.dim_sizes_.push_back(size);
  return out;
}

PartialTensorShape PartialTensorShape::Concatenate(
    const PartialTensorShape& shape) const {
  PartialTensorShape out = *this;
  for (auto s : shape.dim_sizes_) out.dim_sizes_.push_back(s);
  return out;
}

Status PartialTensorShape::MergeWith(const PartialTensorShape& shape,
                                     PartialTensorShape* result) const {
  CHECK(result != this);
  *result = *this;
  if (dims() != shape.dims()) {
    return errors::InvalidArgument(
        "PartialTensorShape: Incompatible ranks during merge: ", dims(),
        " vs. ", shape.dims());
  }
  for (int i = 0; i < dims(); ++i) {
    if (dim_sizes_[i] == -1) {
      result->dim_sizes_[i] = shape.dim_sizes_[i];
    } else if (shape.dim_sizes_[i] != -1 &&
               dim_sizes_[i] != shape.dim_sizes_[i]) {
      return errors::InvalidArgument(
          "PartialTensorShape: Incompatible shapes during merge: ",
          DebugString(), " vs. ", shape.DebugString());
    }
  }
  return Status::OK();
}

void PartialTensorShape::AsProto(TensorShapeProto* proto) const {
  proto->Clear();
  for (size_t d = 0; d < dim_sizes_.size(); ++d) {
    auto* dim = proto->add_dim();
    dim->set_size(dim_sizes_[d]);
  }
}

bool PartialTensorShape::AsTensorShape(TensorShape* shape) const {
  shape->Clear();
  for (auto s : dim_sizes_) {
    if (s < 0) return false;
    shape->AddDim(s);
  }
  return true;
}

string PartialTensorShape::DebugString() const {
  string s = "[";
  bool first = true;
  for (int64 v : dim_sizes_) {
    if (v == -1)
      strings::StrAppend(&s, (first ? "" : ","), "?");
    else
      strings::StrAppend(&s, (first ? "" : ","), v);
    first = false;
  }
  strings::StrAppend(&s, "]");
  return s;
}

string PartialTensorShape::DebugString(const TensorShapeProto& proto) {
  string s = "[";
  bool first = true;
  for (const auto& d : proto.dim()) {
    if (d.size() == -1)
      strings::StrAppend(&s, (first ? "" : ","), "?");
    else
      strings::StrAppend(&s, (first ? "" : ","), d.size());
    first = false;
  }
  strings::StrAppend(&s, "]");
  return s;
}

bool PartialTensorShape::IsCompatibleWith(
    const PartialTensorShape& shape) const {
  if (dims() != shape.dims()) return false;
  for (int i = 0; i < dims(); i++) {
    if (dim_size(i) == -1 || shape.dim_size(i) == -1) continue;
    if (dim_size(i) != shape.dim_size(i)) return false;
  }
  return true;
}

bool PartialTensorShape::IsCompatibleWith(const TensorShape& shape) const {
  if (dims() != shape.dims()) return false;
  for (int i = 0; i < dims(); i++) {
    if (dim_size(i) == -1) continue;
    if (dim_size(i) != shape.dim_size(i)) return false;
  }
  return true;
}

}  // namespace tensorflow
