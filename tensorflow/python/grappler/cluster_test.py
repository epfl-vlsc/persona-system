# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the swig wrapper of clusters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import item
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ClusterTest(test.TestCase):

  def testBasic(self):
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=())
      b = random_ops.random_uniform(shape=())
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      grappler_cluster = cluster.Cluster(
          disable_detailed_stats=False, disable_timeline=False)
      op_perfs, run_time, step_stats = grappler_cluster.MeasureCosts(
          grappler_item)
      self.assertTrue(run_time > 0)
      self.assertEqual(len(op_perfs), 10)
      self.assertTrue(step_stats.dev_stats)

  def testNoDetailedStats(self):
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=())
      b = random_ops.random_uniform(shape=())
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      grappler_cluster = cluster.Cluster(disable_detailed_stats=True)

      op_perfs, run_time, step_stats = grappler_cluster.MeasureCosts(
          grappler_item)
      self.assertTrue(run_time > 0)
      self.assertEqual(len(op_perfs), 0)
      self.assertEqual(len(step_stats.dev_stats), 0)


if __name__ == '__main__':
  test.main()
