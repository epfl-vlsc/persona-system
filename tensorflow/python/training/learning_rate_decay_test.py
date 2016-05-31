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
# ==============================================================================

"""Functional test for learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import learning_rate_decay


class LRDecayTest(test_util.TensorFlowTestCase):

  def testContinuous(self):
    with self.test_session():
      step = 5
      decayed_lr = learning_rate_decay.exponential_decay(0.05, step, 10, 0.96)
      expected = .05 * 0.96 ** (5.0 / 10.0)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testStaircase(self):
    with self.test_session():
      step = state_ops.variable_op([], dtypes.int32)
      assign_100 = state_ops.assign(step, 100)
      assign_1 = state_ops.assign(step, 1)
      assign_2 = state_ops.assign(step, 2)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96**(100 // 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testVariables(self):
    with self.test_session():
      step = variables.Variable(1)
      assign_1 = step.assign(1)
      assign_2 = step.assign(2)
      assign_100 = step.assign(100)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      variables.initialize_all_variables().run()
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96**(100 // 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testPiecewiseConstant(self):
    with self.test_session():
      x = variables.Variable(-999)
      assign_100 = x.assign(100)
      assign_105 = x.assign(105)
      assign_110 = x.assign(110)
      assign_120 = x.assign(120)
      assign_999 = x.assign(999)
      pc = learning_rate_decay.piecewise_constant(x, [100, 110, 120],
                                                  [1.0, 0.1, 0.01, 0.001])
      
      variables.initialize_all_variables().run()
      self.assertAllClose(pc.eval(), 1.0, 1e-6)
      assign_100.op.run()
      self.assertAllClose(pc.eval(), 1.0, 1e-6)
      assign_105.op.run()
      self.assertAllClose(pc.eval(), 0.1, 1e-6)
      assign_110.op.run()
      self.assertAllClose(pc.eval(), 0.1, 1e-6)
      assign_120.op.run()
      self.assertAllClose(pc.eval(), 0.01, 1e-6)
      assign_999.op.run()
      self.assertAllClose(pc.eval(), 0.001, 1e-6)
  
  def testPiecewiseConstantEdgeCases(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        x_int = variables.Variable(0, dtype=variables.dtypes.int32)
        boundaries, values = [-1.0, 1.0], [1, 2, 3]
        pc = learning_rate_decay.piecewise_constant(x_int, boundaries, values)
      with self.assertRaises(ValueError):
        x = variables.Variable(0.0)
        boundaries, values = [-1.0, 1.0], [1.0, 2, 3]
        pc = learning_rate_decay.piecewise_constant(x, boundaries, values)


if __name__ == "__main__":
  googletest.main()
