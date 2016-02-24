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

"""Tests for tensorflow.ops.math_ops.matrix_triangular_solve."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class MatrixTriangularSolveOpTest(tf.test.TestCase):

  def _verifySolve(self, x, y, lower=True, batch_dims=None):
    for np_type in [np.float32, np.float64]:
      a = x.astype(np_type)
      b = y.astype(np_type)
      # For numpy.solve we have to explicitly zero out the strictly
      # upper or lower triangle.
      if lower and a.size > 0:
        a_np = np.tril(a)
      elif a.size > 0:
        a_np = np.triu(a)
      else:
        a_np = a
      if batch_dims is not None:
        a = np.tile(a, batch_dims + [1, 1])
        a_np = np.tile(a_np, batch_dims + [1, 1])
        b = np.tile(b, batch_dims + [1, 1])
      with self.test_session():
        if a.ndim == 2:
          tf_ans = tf.matrix_triangular_solve(a, b, lower=lower).eval()
        else:
          tf_ans = tf.batch_matrix_triangular_solve(a, b, lower=lower).eval()
      np_ans = np.linalg.solve(a_np, b)
      self.assertEqual(np_ans.shape, tf_ans.shape)
      self.assertAllClose(np_ans, tf_ans)

  def testSolve(self):
    # 2x2 matrices, single right-hand side.
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolve(matrix, rhs0, lower=True)
    self._verifySolve(matrix, rhs0, lower=False)
    # 2x2 matrices, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolve(matrix, rhs1, lower=True)
    self._verifySolve(matrix, rhs1, lower=False)

  def testSolveBatch(self):
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolve(matrix, rhs, lower=True, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolve(matrix, rhs, lower=False, batch_dims=[3, 2])

  def testNonSquareMatrix(self):
    # A non-square matrix should cause an error.
    matrix = np.array([[1., 2., 3.], [3., 4., 5.]])
    with self.test_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, matrix)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, matrix, batch_dims=[2, 3])

  def testWrongDimensions(self):
    # The matrix should have the same number of rows as the
    # right-hand sides.
    matrix = np.array([[1., 0.], [0., 1.]])
    rhs = np.array([[1., 0.]])
    with self.test_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

  def testNotInvertible(self):
    # The input should be invertible.
    # The matrix is singular because it has a zero on the diagonal.
    singular_matrix = np.array([[1., 0., -1.], [-1., 0., 1.], [0., -1., 1.]])
    with self.test_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix)
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix, batch_dims=[2, 3])

  def testEmpty(self):
    self._verifySolve(np.empty([0, 2, 2]), np.empty([0, 2, 2]), lower=True)
    self._verifySolve(np.empty([2, 0, 0]), np.empty([2, 0, 0]), lower=True)
    self._verifySolve(np.empty([2, 0, 0]), np.empty([2, 0, 0]), lower=False)
    self._verifySolve(
        np.empty([2, 0, 0]),
        np.empty([2, 0, 0]),
        lower=True,
        batch_dims=[3, 2])


if __name__ == "__main__":
  tf.test.main()
