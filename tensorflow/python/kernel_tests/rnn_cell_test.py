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

"""Tests for RNN cells."""

# pylint: disable=g-bad-import-order,unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class RNNCellTest(tf.test.TestCase):

  def testBasicRNNCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 2])
        g, _ = tf.nn.rnn_cell.BasicRNNCell(2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g], {x.name: np.array([[1., 1.]]),
                             m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))

  def testGRUCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 2])
        g, _ = tf.nn.rnn_cell.GRUCell(2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g], {x.name: np.array([[1., 1.]]),
                             m.name: np.array([[0.1, 0.1]])})
        # Smoke test
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
      with tf.variable_scope("other", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])  # Test GRUCell with input_size != num_units.
        m = tf.zeros([1, 2])
        g, _ = tf.nn.rnn_cell.GRUCell(2, input_size=3)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g], {x.name: np.array([[1., 1., 1.]]),
                             m.name: np.array([[0.1, 0.1]])})
        # Smoke test
        self.assertAllClose(res[0], [[0.156736, 0.156736]])

  def testBasicLSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 8])
        g, out_m = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(2)] * 2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, out_m], {x.name: np.array([[1., 1.]]),
                                    m.name: 0.1 * np.ones([1, 8])})
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        expected_mem = np.array([[0.68967271, 0.68967271,
                                  0.44848421, 0.44848421,
                                  0.39897051, 0.39897051,
                                  0.24024698, 0.24024698]])
        self.assertAllClose(res[1], expected_mem)
      with tf.variable_scope("other", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])  # Test BasicLSTMCell with input_size != num_units.
        m = tf.zeros([1, 4])
        g, out_m = tf.nn.rnn_cell.BasicLSTMCell(2, input_size=3)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, out_m], {x.name: np.array([[1., 1., 1.]]),
                                    m.name: 0.1 * np.ones([1, 4])})
        self.assertEqual(len(res), 2)

  def testLSTMCell(self):
    with self.test_session() as sess:
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([batch_size, input_size])
        m = tf.zeros([batch_size, state_size])
        output, state = tf.nn.rnn_cell.LSTMCell(
            num_units=num_units, input_size=input_size, num_proj=num_proj)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([output, state],
                       {x.name: np.array([[1., 1.], [2., 2.], [3., 3.]]),
                        m.name: 0.1 * np.ones((batch_size, state_size))})
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_proj))
        self.assertEqual(res[1].shape, (batch_size, state_size))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  def testOutputProjectionWrapper(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 3])
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(
            tf.nn.rnn_cell.GRUCell(3), 2)
        g, new_m = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, new_m], {x.name: np.array([[1., 1., 1.]]),
                                    m.name: np.array([[0.1, 0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.231907, 0.231907]])

  def testInputProjectionWrapper(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 3])
        cell = tf.nn.rnn_cell.InputProjectionWrapper(
            tf.nn.rnn_cell.GRUCell(3), 2)
        g, new_m = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, new_m], {x.name: np.array([[1., 1.]]),
                                    m.name: np.array([[0.1, 0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.154605, 0.154605, 0.154605]])

  def testDropoutWrapper(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 3])
        keep = tf.zeros([]) + 1
        g, new_m = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(3),
                                                 keep, keep)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, new_m], {x.name: np.array([[1., 1., 1.]]),
                                    m.name: np.array([[0.1, 0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.154605, 0.154605, 0.154605]])

  def testEmbeddingWrapper(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 1], dtype=tf.int32)
        m = tf.zeros([1, 2])
        g, new_m = tf.nn.rnn_cell.EmbeddingWrapper(
            tf.nn.rnn_cell.GRUCell(2), 3)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, new_m], {x.name: np.array([[1]]),
                                    m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 2))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.17139, 0.17139]])

  def testMultiRNNCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 4])
        _, ml = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(2)] * 2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(ml, {x.name: np.array([[1., 1.]]),
                            m.name: np.array([[0.1, 0.1, 0.1, 0.1]])})
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res, [[0.175991, 0.175991,
                                   0.13248, 0.13248]])


if __name__ == "__main__":
  tf.test.main()
