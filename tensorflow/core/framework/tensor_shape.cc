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

#include "tensorflow/core/public/tensor_shape.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// An upper limit of the total number of elements in a tensor.
static const int64 kMaxElements = (1LL << 40);

bool TensorShape::IsValid(const TensorShapeProto& proto) {
  int64 num_elements = 1;
  for (const auto& d : proto.dim()) {
    if (d.size() < 0) return false;
    num_elements *= d.size();
    if (num_elements > kMaxElements) return false;
  }
  return true;
}

Status TensorShape::IsValidShape(const TensorShapeProto& proto) {
  int64 num_elements = 1;
  for (const auto& d : proto.dim()) {
    if (d.size() < 0) {
      return errors::InvalidArgument("Shape ", DebugString(proto),
                                     " has negative dimensions");
    }
    num_elements *= d.size();
    if (num_elements > kMaxElements) {
      return errors::InvalidArgument("Shape ", DebugString(proto),
                                     " is too large (more than ", kMaxElements,
                                     " entries)");
    }
  }
  return Status::OK();
}

TensorShape::TensorShape(const TensorShapeProto& proto) {
  dim_sizes_.reserve(proto.dim_size());
  num_elements_ = 1;
  for (const auto& d : proto.dim()) {
    AddDim(d.size());
  }
}

TensorShape::TensorShape(gtl::ArraySlice<int64> dim_sizes) {
  dim_sizes_.reserve(dim_sizes.size());
  num_elements_ = 1;
  for (auto s : dim_sizes) {
    AddDim(s);
  }
}

TensorShape::TensorShape() : num_elements_(1) {}

void TensorShape::Clear() {
  dim_sizes_.clear();
  num_elements_ = 1;
}

void TensorShape::AddDim(int64 size) {
  CHECK_GE(size, 0);
  dim_sizes_.push_back(size);
  num_elements_ *= size;
  CHECK_LE(0, num_elements_);
  CHECK_LE(num_elements_, kMaxElements);
}

void TensorShape::AppendShape(const TensorShape& shape) {
  for (auto d : shape) AddDim(d.size);
}

void TensorShape::InsertDim(int d, int64 size) {
  CHECK_GE(d, 0);
  CHECK_LE(d, dims());
  CHECK_GE(size, 0);
  dim_sizes_.insert(dim_sizes_.begin() + d, size);
  num_elements_ *= size;
  CHECK_LE(0, num_elements_);
  CHECK_LE(num_elements_, kMaxElements);
}

void TensorShape::set_dim(int d, int64 size) {
  CHECK_GE(d, 0);
  CHECK_LT(d, dims());
  CHECK_GE(size, 0);

  // Update the number of elements. num_elements_ is int64.
  dim_sizes_[d] = size;
  recompute_dims();
}

void TensorShape::RemoveDim(int d) {
  CHECK_GE(d, 0);
  CHECK_LT(d, dims());

  // Update the number of elements and remove the dimension from the
  // sizes.
  dim_sizes_.erase(dim_sizes_.begin() + d);
  recompute_dims();
}

void TensorShape::recompute_dims() {
  num_elements_ = 1;
  for (auto s : dim_sizes_) {
    num_elements_ *= s;
    CHECK_LE(0, num_elements_);
    CHECK_LE(num_elements_, kMaxElements);
  }
}

bool TensorShape::IsSameSize(const TensorShape& b) const {
  if (b.dims() != dims()) return false;
  for (int d = 0; d < dims(); d++) {
    if (dim_size(d) != b.dim_size(d)) return false;
  }
  return true;
}

void TensorShape::AsProto(TensorShapeProto* proto) const {
  proto->Clear();
  for (size_t d = 0; d < dim_sizes_.size(); ++d) {
    auto* dim = proto->add_dim();
    dim->set_size(dim_sizes_[d]);
  }
}

TensorShapeIter TensorShape::begin() const { return TensorShapeIter(this, 0); }

TensorShapeIter TensorShape::end() const {
  return TensorShapeIter(this, dims());
}

string TensorShape::DebugString() const {
  return strings::StrCat(
      "[", str_util::Join(gtl::ArraySlice<int64>(dim_sizes_), ","), "]");
}

string TensorShape::DebugString(const TensorShapeProto& proto) {
  string s = "[";
  bool first = true;
  for (const auto& d : proto.dim()) {
    strings::StrAppend(&s, first ? "" : ",", d.size());
    first = false;
  }
  strings::StrAppend(&s, "]");
  return s;
}

bool TensorShapeUtils::StartsWith(const TensorShape& shape,
                                  const TensorShape& prefix) {
  if (shape.dims() < prefix.dims()) return false;
  for (int i = 0; i < prefix.dims(); i++) {
    if (shape.dim_size(i) != prefix.dim_size(i)) return false;
  }
  return true;
}

}  // namespace tensorflow
