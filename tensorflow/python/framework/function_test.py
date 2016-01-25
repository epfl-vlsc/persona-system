# Copyright 2015 Google Inc. All Rights Reserved.
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
# =============================================================================
"""Tests for functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
# pylint: enable=unused-import,g-bad-import-order

import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import function
# pylint: disable=unused-import
from tensorflow.python.ops import functional_ops

# pylint: enable=unused-import


class FunctionTest(tf.test.TestCase):

  def _mat(self, x):
    return np.array([x]).astype("float32").reshape([1, 1])

  def testBasic(self):
    g = tf.Graph()

    # Define a function
    #   foo(a:float, b:float, c:float)->u:float,v:float,w:float
    #     u = matmul(a, b) + c
    #     v = u^2
    #     w = u + v
    # TODO(zhifengc): replaces w/ a nicer @decorator sugar.
    foo = tf.Graph()
    with foo.as_default():
      a = tf.placeholder(tf.float32, name="a")
      b = tf.placeholder(tf.float32, name="b")
      c = tf.placeholder(tf.float32, name="c")
      u = tf.add(tf.matmul(a, b), c, name="u")
      v = tf.square(u, name="v")
      w = tf.add_n([u, v], name="w")
    fdef = function.graph_to_function_def(foo, "foo", [a, b, c], [u, v, w])
    g._add_function(fdef)

    # Compute 2 * 3 + 4 and its square.
    with g.as_default(), tf.Session() as sess:
      two = tf.constant(self._mat(2.0), name="two")
      three = tf.constant(self._mat(3.0), name="three")
      four = tf.constant(self._mat(4.0), name="four")
      # TODO(zhifengc): w/ @decorator sugar, we will just do:
      #   y, s, t = foo_func(two, three, four)

      # The graph contains two ops each of which calls foo.
      u0, v0, w0 = g.create_op("foo",
                               [two, three, four],
                               [tf.float32, tf.float32, tf.float32],
                               compute_shapes=False).outputs
      u1, v1, w1 = g.create_op("foo",
                               [four, two, three],
                               [tf.float32, tf.float32, tf.float32],
                               compute_shapes=False).outputs

      # Checks some property of the graph def.
      gdef = g.as_graph_def()
      self.assertEqual(len(gdef.node), 5)  # 5 nodes added.
      self.assertEqual(len(gdef.library.function), 1)  # 1 function is defined.

      for _ in xrange(10):
        # Run the graph, which is basicly two function calls.
        ans_u0, ans_v0, ans_w0, ans_u1, ans_v1, ans_w1 = sess.run([u0, v0, w0,
                                                                   u1, v1, w1])
        self.assertAllEqual(ans_u0, self._mat(10.0))  # 2 * 3 + 4 = 10
        self.assertAllEqual(ans_v0, self._mat(100.0))  # 10^2 = 100
        self.assertAllEqual(ans_w0, self._mat(110.0))  # 100 + 10 = 110
        self.assertAllEqual(ans_u1, self._mat(11.0))  # 4 * 2 + 3 = 11
        self.assertAllEqual(ans_v1, self._mat(121.0))  # 11^2 = 121
        self.assertAllEqual(ans_w1, self._mat(132.0))  # 11 + 121 = 132

  def testDefineFunction2Args(self):

    def APlus2B(a, b):
      return a + b * 2

    with tf.Graph().as_default():
      f_def = function.define_function(APlus2B, {"a": tf.float32,
                                                 "b": tf.float32})
      one = tf.constant([1.0])
      two = tf.constant([2.0])
      call = function.call_function(f_def, one, two)
      self.assertEquals("APlus2B", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([5.0], sess.run(call))

  def testGradientFunc(self):

    def XSquarePlusOne(x):
      return x * x + 1.0

    def XSquarePlusOneGrad(x, dy):
      dx = functional_ops._symbolic_gradient(input=[x, dy],
                                             Tout=[tf.float32],
                                             f="XSquarePlusOne",
                                             name="dx")
      return dx

    g = tf.Graph()
    with g.as_default():
      f = function.define_function(XSquarePlusOne, {"x": tf.float32})
      g = function.define_function(XSquarePlusOneGrad, {"x": tf.float32,
                                                        "dy": tf.float32})
      epsilon = tf.constant([0.1])
      two = tf.constant([2.0])
      call_f = function.call_function(f, two)
      call_g = function.call_function(g, two, epsilon)

      with tf.Session() as sess:
        self.assertAllClose([5.0], sess.run(call_f))
        self.assertAllClose([0.4], sess.run(call_g))

  def testSymGradShape(self):
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, [25, 4])
      y = tf.placeholder(tf.float32, [200, 100])
      dz = tf.placeholder(tf.float32, [1])
      # We assume Foo is a function of (x, y) -> (z) Then, Foo's
      # gradient function is (x, y, dz) -> (dx, dy).  dx's shape
      # should be the same as x's; and dy's shape should be the same
      # as y's.
      dx, dy = functional_ops._symbolic_gradient(input=[x, y, dz],
                                                 Tout=[tf.float32] * 2,
                                                 f="Foo")
      self.assertEquals(x.get_shape(), dx.get_shape())
      self.assertEquals(y.get_shape(), dy.get_shape())

  def testDefineFunctionNoArgs(self):

    def AConstant():
      return tf.constant([42])

    with tf.Graph().as_default():
      f_def = function.define_function(AConstant, {})
      call = function.call_function(f_def)
      self.assertEquals("AConstant", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([42], sess.run(call))

  def testDefineFunctionNames(self):

    def Foo(a):
      return a + 1

    with tf.Graph().as_default():
      f_def = function.define_function(Foo, {"a": tf.float32})
      one = tf.constant([1.0])
      call1 = function.call_function(f_def, one)
      self.assertEquals("Foo", call1.op.name)
      call2 = function.call_function(f_def, one)
      self.assertEquals("Foo_1", call2.op.name)
      call3 = function.call_function(f_def, one, name="mine")
      self.assertEquals("mine", call3.op.name)
      with tf.name_scope("my"):
        call4 = function.call_function(f_def, one, name="precious")
        self.assertEquals("my/precious", call4.op.name)

  def testDefineErrors(self):

    def NoResult():
      pass

    def VarArgs(*unused_b):
      return tf.constant([1])

    def DefaultArg(unused_a=12):
      return tf.constant([1])

    def KwArgs(**unused_kwargs):
      return tf.constant([1])

    def PlusMinus(a, b):
      return a + b, b - a

    with tf.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "return at least one tensor"):
        function.define_function(NoResult, {})
      with self.assertRaisesRegexp(ValueError, "plain arglists are supported"):
        function.define_function(VarArgs, {})
      with self.assertRaisesRegexp(ValueError, "plain arglists are supported"):
        function.define_function(DefaultArg, {})
      with self.assertRaisesRegexp(ValueError, "plain arglists are supported"):
        function.define_function(KwArgs, {})
      with self.assertRaisesRegexp(ValueError, "specified input types"):
        function.define_function(PlusMinus, {})
      with self.assertRaisesRegexp(ValueError, "specified input types"):
        function.define_function(PlusMinus, {"c": tf.float32})
      with self.assertRaisesRegexp(ValueError, "type for argument: b"):
        function.define_function(PlusMinus, {"a": tf.float32,
                                             "c": tf.float32})
      with self.assertRaisesRegexp(ValueError, "specified input types"):
        function.define_function(PlusMinus, {"a": tf.float32,
                                             "b": tf.float32,
                                             "c": tf.float32})

  def testCallErrors(self):

    def Const():
      return tf.constant(1)

    def PlusOne(a):
      return a + 1

    def PlusMinus(a, b):
      return a + b, b - a

    with tf.Graph().as_default():
      one = tf.constant([1])
      two = tf.constant([2])
      const = function.define_function(Const, {})
      plus_one = function.define_function(PlusOne, {"a": tf.int32})
      plus_minus = function.define_function(PlusMinus, {"a": tf.int32,
                                                        "b": tf.int32})

      function.call_function(const)
      with self.assertRaisesRegexp(ValueError, "arguments: 0"):
        function.call_function(const, one)
      with self.assertRaisesRegexp(ValueError, "arguments: 0"):
        function.call_function(const, one, two)

      with self.assertRaisesRegexp(ValueError, "arguments: 1"):
        function.call_function(plus_one)
      function.call_function(plus_one, one)
      with self.assertRaisesRegexp(ValueError, "arguments: 1"):
        function.call_function(plus_one, one, two)

      with self.assertRaisesRegexp(ValueError, "arguments: 2"):
        function.call_function(plus_minus)
      with self.assertRaisesRegexp(ValueError, "arguments: 2"):
        function.call_function(plus_minus, one)
      function.call_function(plus_minus, one, two)

      function.call_function(plus_one, one, name="p1")
      with self.assertRaisesRegexp(ValueError, "Unknown keyword arguments"):
        function.call_function(plus_one, one, device="/gpu:0")

  def testFunctionDecorator(self):

    with tf.Graph().as_default():

      @function.Defun(b=tf.int32)
      def Minus1(b):
        return b - 1

      two = tf.constant([2])
      call1 = Minus1(two)
      self.assertEquals("Minus1", call1.op.name)
      # pylint: disable=unexpected-keyword-arg
      call2 = Minus1(call1, name="next")
      # pylint:enable=unexpected-keyword-arg
      self.assertEquals("next", call2.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([1], sess.run(call1))
        self.assertAllEqual([0], sess.run(call2))

  def testNestedFunction(self):
    with tf.Graph().as_default():

      @function.Defun(x=tf.float32)
      def Cube(x):
        return x * x * x

      @function.Defun(x=tf.float32, y=tf.float32)
      def CubeXPlusY(x, y):
        return Cube(x) + y

      z = CubeXPlusY(tf.constant(3.0), tf.constant(-2.0))
      with self.test_session():
        self.assertAllEqual(z.eval(), 25.0)


class UnrollLSTMTest(tf.test.TestCase):
  BATCH_SIZE = 16
  LSTM_DIMS = 32
  NUM_UNROLL = 100

  def _Weights(self):
    dims = self.LSTM_DIMS
    return tf.random_uniform([2 * dims, 4 * dims], -1, 1, seed=123456)

  def _Input(self):
    return tf.random_uniform(
        [self.NUM_UNROLL, self.BATCH_SIZE, self.LSTM_DIMS],
        seed=654321)

  # Helper to construct a LSTM cell graph.
  @classmethod
  def LSTMCell(cls, x, mprev, cprev, weights):
    xm = tf.concat(1, [x, mprev])
    i_i, i_g, f_g, o_g = tf.split(1, 4, tf.matmul(xm, weights))
    new_c = tf.sigmoid(f_g) * cprev + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c = tf.clip_by_value(new_c, -50.0, 50.0)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):

    def Loop(cell, w, i):
      x = tf.unpack(i, self.NUM_UNROLL)
      m = tf.zeros_like(x[0])
      c = tf.zeros_like(x[0])
      for i in range(self.NUM_UNROLL):
        m, c = cell(x[i], m, c, w)
      return m

    cell = UnrollLSTMTest.LSTMCell
    if mode == "complete":
      # Constructs the complete graph in python.
      return Loop(cell, weights, inp)

    cell = function.Defun(x=tf.float32,
                          mprev=tf.float32,
                          cprev=tf.float32,
                          weights=tf.float32)(cell)
    if mode == "cell":
      # Just represent the LSTM as a function.
      return Loop(cell, weights, inp)

    if mode == "loop":
      # Wraps the whole loop as a function.
      @function.Defun(w=tf.float32, i=tf.float32)
      def LSTMLoop(w, i):
        return Loop(cell, w, i)

      return LSTMLoop(weights, inp)

    if mode == "loop10":
      # Wraps 10 lstm steps into one function, and the whole loop
      # into another calling the formers.

      # Groups 10 steps at a time):
      # TODO(zhifengc): Any way to make the syntax less hideous?
      @function.Defun(m=tf.float32,
                      c=tf.float32,
                      w=tf.float32,
                      x0=tf.float32,
                      x1=tf.float32,
                      x2=tf.float32,
                      x3=tf.float32,
                      x4=tf.float32,
                      x5=tf.float32,
                      x6=tf.float32,
                      x7=tf.float32,
                      x8=tf.float32,
                      x9=tf.float32)
      def Loop10(w, m, c, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        for x in [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]:
          m, c = cell(x, m, c, w)
        return m, c

      @function.Defun(weights=tf.float32, inp=tf.float32)
      def LSTMLoop10(weights, inp):
        x = tf.unpack(inp, self.NUM_UNROLL)
        m = tf.zeros_like(x[0])
        c = tf.zeros_like(x[0])
        assert self.NUM_UNROLL % 10 == 0
        for i in range(0, self.NUM_UNROLL, 10):
          m, c = Loop10(weights, m, c, *x[i:i + 10])
        return m

      return LSTMLoop10(weights, inp)

  def testUnrollLSTM(self):
    # Run one step of the unrolled lstm graph.
    def RunForward(mode):
      g = tf.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
      gdef = g.as_graph_def()
      finish = time.time()
      print("time: ", finish - start, " txt size: ", len(str(gdef)),
            "gdef bin size: ", len(gdef.SerializeToString()))
      with g.as_default(), tf.Session() as sess:
        return sess.run(m)

    mv0 = RunForward("complete")
    mv1 = RunForward("cell")
    mv2 = RunForward("loop")
    mv3 = RunForward("loop10")
    self.assertAllClose(mv0, mv1, rtol=1e-4)
    self.assertAllClose(mv0, mv2, rtol=1e-4)
    self.assertAllClose(mv0, mv3, rtol=1e-4)

  def testUnrollLSTMGrad(self):
    # Run one step of the unrolled lstm graph.
    def RunForwardBackward(mode):
      g = tf.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
        loss = tf.reduce_sum(tf.square(m))
        dw = tf.gradients([loss], [weights])
      gdef = g.as_graph_def()
      finish = time.time()
      print("time: ", finish - start, " txt size: ", len(str(gdef)),
            "gdef bin size: ", len(gdef.SerializeToString()))
      with g.as_default(), tf.Session() as sess:
        return sess.run(dw)

    d0 = RunForwardBackward("complete")
    d1 = RunForwardBackward("cell")
    d2 = RunForwardBackward("loop")
    d3 = RunForwardBackward("loop10")
    self.assertAllClose(d0, d1, rtol=1e-4)
    self.assertAllClose(d0, d2, rtol=1e-4)
    self.assertAllClose(d0, d3, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
