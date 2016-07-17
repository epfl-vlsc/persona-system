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

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

TEST(ShapeInferenceTest, RankAndDimInspection) {
  InferenceContext c({"?", "[1,?,3]", "[]"}, 2 /* num_outputs */);
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(2, c.num_outputs());

  auto in0 = c.input(0);
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_FALSE(c.RankKnown(in0));
  EXPECT_EQ(InferenceContext::kUnknownRank, c.Rank(in0));

  auto in1 = c.input(1);
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
  EXPECT_TRUE(c.RankKnown(in1));
  EXPECT_EQ(3, c.Rank(in1));
  auto d = c.Dim(in1, 0);
  EXPECT_EQ(1, c.Value(d));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("1", c.DebugString(d));
  d = c.Dim(in1, 1);
  EXPECT_EQ(InferenceContext::kUnknownDim, c.Value(d));
  EXPECT_FALSE(c.ValueKnown(d));
  EXPECT_EQ("?", c.DebugString(d));
  d = c.Dim(in1, 2);
  EXPECT_EQ(3, c.Value(d));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("3", c.DebugString(d));

  auto in2 = c.input(2);
  EXPECT_EQ("[]", c.DebugString(in2));
  EXPECT_TRUE(c.RankKnown(in2));
  EXPECT_EQ(0, c.Rank(in2));
}

TEST(ShapeInferenceTest, WithRank) {
  InferenceContext c({"?", "[1,?,3]"}, 2 /* num_outputs */);

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  const Shape* s1 = nullptr;
  const Shape* s2 = nullptr;

  // WithRank on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRank(in0, 1, &s1).ok());
  EXPECT_EQ("[?]", c.DebugString(s1));

  EXPECT_TRUE(c.WithRank(in0, 2, &s2).ok());
  EXPECT_EQ("[?,?]", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);                      // different pointers
  EXPECT_TRUE(c.Dim(s2, 0) != c.Dim(s2, 1));  // different pointers.

  EXPECT_TRUE(c.WithRank(in0, 1, &s2).ok());
  EXPECT_EQ("[?]", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);  // different pointers

  EXPECT_TRUE(c.WithRank(in0, 0, &s1).ok());
  EXPECT_EQ("[]", c.DebugString(s1));

  // WithRank on shape with known dimensionality.
  s1 = in1;
  EXPECT_EQ("Invalid argument: Shape must be rank 2 but is rank 3",
            c.WithRank(in1, 2, &s1).ToString());
  EXPECT_TRUE(s1 == nullptr);
  EXPECT_TRUE(c.WithRank(in1, 3, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST(ShapeInferenceTest, WithValue) {
  InferenceContext c({"[1,?]"}, 2 /* num_outputs */);

  auto d0 = c.Dim(c.input(0), 0);
  auto d1 = c.Dim(c.input(0), 1);
  const Dimension* out1 = nullptr;
  const Dimension* out2 = nullptr;

  // WithRank on a dimension with unknown value always succeeds.
  EXPECT_TRUE(c.WithValue(d1, 1, &out1).ok());
  EXPECT_EQ(1, c.Value(out1));

  EXPECT_TRUE(c.WithValue(d1, 2, &out2).ok());
  EXPECT_EQ(2, c.Value(out2));
  EXPECT_TRUE(out1 != out2);  // different pointers
  EXPECT_TRUE(out1 != d1);    // different pointers

  EXPECT_TRUE(c.WithValue(d1, 1, &out2).ok());
  EXPECT_EQ(1, c.Value(out2));
  EXPECT_TRUE(out1 != out2);  // different pointers

  // WithRank on dimension with known size.
  out1 = d0;
  EXPECT_EQ("Invalid argument: Dimension must be size 0 but is size 1",
            c.WithValue(d0, 0, &out1).ToString());
  EXPECT_TRUE(out1 == nullptr);
  out1 = d0;
  EXPECT_EQ("Invalid argument: Dimension must be size 2 but is size 1",
            c.WithValue(d0, 2, &out1).ToString());
  EXPECT_TRUE(out1 == nullptr);
  EXPECT_TRUE(c.WithValue(d0, 1, &out1).ok());
  EXPECT_TRUE(d0 == out1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
}

TEST(ShapeInferenceTest, MergeDim) {
  InferenceContext c({"[2,?,2,1,?]"}, 2 /* num_outputs */);

  auto d2 = c.Dim(c.input(0), 0);
  auto d_unknown = c.Dim(c.input(0), 1);
  auto d2_b = c.Dim(c.input(0), 2);
  auto d1 = c.Dim(c.input(0), 3);
  auto d_unknown_b = c.Dim(c.input(0), 4);
  const Dimension* out = nullptr;

  // Merging anything with unknown returns the same pointer.
  EXPECT_TRUE(c.Merge(d2, d_unknown, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d2, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown_b, &out).ok());
  EXPECT_TRUE(d_unknown == out);

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(d2, d2, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown, &out).ok());
  EXPECT_TRUE(d_unknown == out);

  // Merging equal values returns first one.
  EXPECT_TRUE(c.Merge(d2, d2_b, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d2_b, d2, &out).ok());
  EXPECT_TRUE(d2_b == out);

  // Merging inequal values is an error.
  EXPECT_EQ("Invalid argument: Dimensions must be equal size, but are 2 and 1",
            c.Merge(d2, d1, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  EXPECT_EQ("Invalid argument: Dimensions must be equal size, but are 1 and 2",
            c.Merge(d1, d2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
}

TEST(ShapeInferenceTest, MergeShape) {
  InferenceContext c({"?", "[1,2]", "[?,2]", "[1,?]", "[1,3]", "?", "[1]"},
                     2 /* num_outputs */);

  auto s_unknown = c.input(0);
  auto s_1_2 = c.input(1);
  auto s_u_2 = c.input(2);
  auto s_1_u = c.input(3);
  auto s_1_3 = c.input(4);
  auto s_unknown_b = c.input(5);
  auto s_1 = c.input(6);
  const Shape* out = nullptr;

  // Merging any shape with unknown returns the shape.
  EXPECT_TRUE(c.Merge(s_unknown, s_1_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);
  EXPECT_TRUE(c.Merge(s_u_2, s_unknown, &out).ok());
  EXPECT_TRUE(s_u_2 == out);
  EXPECT_TRUE(c.Merge(s_unknown, s_unknown_b, &out).ok());
  EXPECT_TRUE(s_unknown == out);

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(s_1_2, s_1_2, &out).ok());
  EXPECT_TRUE(out == s_1_2);

  // Merging where one of the inputs is the right answer - return that input.
  out = nullptr;
  EXPECT_TRUE(c.Merge(s_1_2, s_u_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);
  out = nullptr;
  EXPECT_TRUE(c.Merge(s_u_2, s_1_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);

  // Merging where neither input is the right answer.
  EXPECT_TRUE(c.Merge(s_u_2, s_1_u, &out).ok());
  EXPECT_TRUE(out != s_u_2);
  EXPECT_TRUE(out != s_1_u);
  EXPECT_EQ("[1,2]", c.DebugString(out));
  EXPECT_TRUE(c.Dim(s_1_u, 0) == c.Dim(out, 0));  // same pointers
  EXPECT_TRUE(c.Dim(s_u_2, 1) == c.Dim(out, 1));  // same pointers

  // Incompatible merges give errors and set out to nullptr.
  out = s_unknown;
  EXPECT_EQ("Invalid argument: Dimensions must be equal size, but are 2 and 3",
            c.Merge(s_u_2, s_1_3, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = s_unknown;
  EXPECT_EQ("Invalid argument: Dimensions must be equal size, but are 3 and 2",
            c.Merge(s_1_3, s_u_2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = s_unknown;
  EXPECT_EQ("Invalid argument: Shapes must be equal rank, but are 1 and 2",
            c.Merge(s_1, s_1_2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
}

TEST(ShapeInferenceTest, CreateShape) {
  InferenceContext c({"[1,2,3,?,5]"}, 2 /* num_outputs */);

  std::vector<const Dimension*> dims;
  auto in0 = c.input(0);
  const int rank = c.Rank(in0);
  for (int i = 0; i < rank; ++i) {
    dims.push_back(c.Dim(in0, rank - i - 1));
  }

  auto s = c.CreateShape(dims);
  EXPECT_EQ("[5,?,3,2,1]", c.DebugString(s));
  EXPECT_TRUE(c.Dim(s, 0) == c.Dim(in0, rank - 1));

  auto s2 = c.CreateShape(dims);
  EXPECT_TRUE(s != s2);  // different pointers
  EXPECT_TRUE(c.Dim(s2, 0) == c.Dim(in0, rank - 1));
}

TEST(ShapeInferenceTest, CreateUnknownShape) {
  InferenceContext c({}, 2 /* num_outputs */);

  auto u0 = c.CreateUnknownShape();
  auto u1 = c.CreateUnknownShape();
  EXPECT_EQ("?", c.DebugString(u0));
  EXPECT_EQ("?", c.DebugString(u1));
  EXPECT_TRUE(u0 != u1);  // different pointers
}

TEST(ShapeInferenceTest, CreateDim) {
  InferenceContext c({}, 2 /* num_outputs */);

  auto* d0 = c.CreateDim(1);
  auto* d1 = c.CreateDim(1);
  auto* d2 = c.CreateDim(2);
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("1", c.DebugString(d1));
  EXPECT_TRUE(d0 != d1);  // different pointers
  EXPECT_EQ("2", c.DebugString(d2));
}

TEST(ShapeInferenceTest, CreateUnknownDim) {
  InferenceContext c({}, 2 /* num_outputs */);

  auto* d0 = c.CreateUnknownDim();
  auto* d1 = c.CreateUnknownDim();
  EXPECT_EQ("?", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
  EXPECT_TRUE(d0 != d1);  // different pointers
}

}  // namespace shape_inference
}  // namespace tensorflow
