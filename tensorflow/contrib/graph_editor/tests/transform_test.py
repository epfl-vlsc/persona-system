# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.contrib.graph_editor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


class TransformTest(tf.test.TestCase):

  def setUp(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      c0 = tf.constant(1.0, shape=[10], name="Const")
      c1 = tf.constant(1.0, shape=[10], name="Const")
      c2 = tf.constant(1.0, shape=[10], name="Const")
      i = tf.constant(1.0, shape=[10], name="Input")
      self.o = tf.add(c2, tf.add(c1, tf.add(c0, i)))

  def test_copy(self):
    graph = tf.Graph()
    ge.copy(self.graph, graph)
    self.assertEqual(set(op.name for op in self.graph.get_operations()),
                     set(op.name for op in graph.get_operations()))

  def test_transform(self):
    transformer = ge.Transformer()
    def my_transform_op_handler(info, op):
      add_noise = op.name.startswith("Add")
      op_ = ge.transform.copy_op_handler(info, op)
      if add_noise:
        # add some noise to op
        with info.graph_.as_default():
          t_ = tf.add(tf.constant(1.0, shape=[10], name="Noise"),
                      op_.outputs[0], name="AddNoise")
        # return the "noisy" op
        return t_.op
      else:
        return op_
    transformer.transform_op_handler = my_transform_op_handler

    graph = tf.Graph()
    transformer(self.graph, graph, "", "")
    matcher0 = ge.matcher("AddNoise").input_ops(
        "Noise", ge.matcher("Add").input_ops("Const", "Input"))
    matcher1 = ge.matcher("AddNoise_1").input_ops(
        "Noise_1", ge.matcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = ge.matcher("AddNoise_2").input_ops(
        "Noise_2", ge.matcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=graph)[0]
    self.assertTrue(matcher2(top))

  def test_transform_in_place(self):
    transformer = ge.Transformer()
    def my_transform_op_handler_in_place(info, op):
      add_noise = op.name.startswith("Add")
      op = ge.transform.transform_op_in_place(info, op,
                                              detach_outputs=add_noise)
      if add_noise:
        # add some noise to op
        with info.graph_.as_default():
          t = tf.add(tf.constant(1.0, shape=[10], name="Noise"), op.outputs[0],
                     name="AddNoise")
        # return the "noisy" op
        return t.op
      else:
        return op
    transformer.transform_op_handler = my_transform_op_handler_in_place

    transformer(self.graph, self.graph, "", "")
    matcher0 = ge.matcher("AddNoise").input_ops(
        "Noise", ge.matcher("Add").input_ops("Const", "Input"))
    matcher1 = ge.matcher("AddNoise_1").input_ops(
        "Noise_1", ge.matcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = ge.matcher("AddNoise_2").input_ops(
        "Noise_2", ge.matcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=self.graph)[0]
    self.assertTrue(matcher2(top))

if __name__ == "__main__":
  tf.test.main()
