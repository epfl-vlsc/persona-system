/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace shape_inference {

constexpr int32 InferenceContext::kUnknownRank;
constexpr int64 InferenceContext::kUnknownDim;

InferenceContext::InferenceContext(const std::vector<string>& input_shapes,
                                   int num_outputs) {
  for (const string& spec : input_shapes) {
    if (spec == "?") {
      inputs_.push_back(CreateUnknownShape());
    } else {
      std::vector<const Dimension*> dims;
      strings::Scanner scanner(spec);
      scanner.OneLiteral("[");
      while (scanner.Peek() != ']') {
        if (scanner.Peek() == '?') {
          scanner.OneLiteral("?");
          dims.push_back(CreateUnknownDim());
        } else {
          scanner.RestartCapture().Many(strings::Scanner::DIGIT);
          StringPiece match;
          int64 dim_size = 0;
          CHECK(scanner.GetResult(nullptr, &match) &&
                strings::safe_strto64(match, &dim_size))
              << spec;
          dims.push_back(CreateDim(dim_size));
        }

        if (scanner.Peek() == ',') {
          scanner.OneLiteral(",");
        } else {
          CHECK_EQ(scanner.Peek(), ']');
        }
      }
      CHECK(scanner.OneLiteral("]").Eos().GetResult()) << spec;
      inputs_.push_back(CreateShape(dims));
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    outputs_.push_back(CreateUnknownShape());
  }
}

InferenceContext::~InferenceContext() {
  for (auto* s : all_shapes_) delete s;
  for (auto* d : all_dims_) delete d;
}

string InferenceContext::DebugString(const Shape* s) {
  if (RankKnown(s)) {
    std::vector<string> vals;
    for (auto d : s->dims_) vals.push_back(DebugString(d));
    return strings::StrCat("[", str_util::Join(vals, ","), "]");
  } else {
    return "?";
  }
}

string InferenceContext::DebugString(const Dimension* d) {
  return ValueKnown(d) ? strings::StrCat(Value(d)) : "?";
}

// If <shape> has rank <rank>, or its rank is unknown, return OK and return
// the shape with asserted rank in <*out>. Otherwise return an error.
Status InferenceContext::WithRank(const Shape* shape, int32 rank,
                                  const Shape** out) {
  const int32 existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<const Dimension*> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      all_dims_.push_back(new Dimension());
      dims.push_back(all_dims_.back());
    }
    all_shapes_.push_back(new Shape(dims));
    *out = all_shapes_.back();
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be rank ", rank, " but is rank ",
                                 existing);
}

Status InferenceContext::WithValue(const Dimension* dim, int64 value,
                                   const Dimension** out) {
  const int64 existing = Value(dim);
  if (existing == value) {
    *out = dim;
    return Status::OK();
  }
  if (existing == kUnknownDim) {
    all_dims_.push_back(new Dimension(value));
    *out = all_dims_.back();
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Dimension must be size ", value,
                                 " but is size ", existing);
}

Status InferenceContext::Merge(const Dimension* d0, const Dimension* d1,
                               const Dimension** out) {
  if (d0 == d1 || !ValueKnown(d1)) {
    *out = d0;
    return Status::OK();
  } else if (!ValueKnown(d0)) {
    *out = d1;
    return Status::OK();
  } else if (Value(d0) == Value(d1)) {
    *out = d0;
    return Status::OK();
  } else {
    *out = nullptr;
    return errors::InvalidArgument("Dimensions must be equal size, but are ",
                                   Value(d0), " and ", Value(d1));
  }
}

Status InferenceContext::Merge(const Shape* s0, const Shape* s1,
                               const Shape** out) {
  if (s0 == s1 || !RankKnown(s1)) {
    *out = s0;
    return Status::OK();
  } else if (!RankKnown(s0)) {
    *out = s1;
    return Status::OK();
  }

  const int32 rank = Rank(s0);
  if (rank != Rank(s1)) {
    *out = nullptr;
    return errors::InvalidArgument("Shapes must be equal rank, but are ", rank,
                                   " and ", Rank(s1));
  }

  bool return_s0 = true;
  bool return_s1 = true;
  for (int i = 0; i < rank; ++i) {
    auto d0 = Dim(s0, i);
    auto d1 = Dim(s1, i);
    if (d0 == d1) continue;

    auto v0 = Value(d0);
    auto v1 = Value(d1);
    if (v0 == kUnknownDim) {
      if (v1 != kUnknownDim) {
        return_s0 = false;
      }
    } else if (v1 == kUnknownDim) {
      return_s1 = false;
    } else if (v0 != v1) {
      *out = nullptr;
      return errors::InvalidArgument("Dimensions must be equal size, but are ",
                                     Value(d0), " and ", Value(d1));
    }
  }
  if (return_s0 || return_s1) {
    *out = return_s0 ? s0 : s1;
    return Status::OK();
  }

  // Merge dims.
  std::vector<const Dimension*> dims(rank, nullptr);
  for (int i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    TF_CHECK_OK(Merge(Dim(s0, i), Dim(s1, i), &dims[i]));
  }
  *out = CreateShape(dims);
  return Status::OK();
}

const Shape* InferenceContext::CreateShape(
    const std::vector<const Dimension*>& dims) {
  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();
}

const Shape* InferenceContext::CreateUnknownShape() {
  all_shapes_.push_back(new Shape());
  return all_shapes_.back();
}

const Dimension* InferenceContext::CreateDim(int64 value) {
  all_dims_.push_back(new Dimension(value));
  return all_dims_.back();
}

const Dimension* InferenceContext::CreateUnknownDim() {
  all_dims_.push_back(new Dimension());
  return all_dims_.back();
}

}  // namespace shape_inference
}  // namespace tensorflow
