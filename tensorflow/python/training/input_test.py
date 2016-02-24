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

"""Tests for training.input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class MatchFilenamesOnceTest(tf.test.TestCase):

  def test(self):
    temp_dir = self.get_temp_dir()
    filenames = [os.path.join(temp_dir, n) for n in os.listdir(temp_dir)]
    additional = [os.path.join(self.get_temp_dir(), "match_filenames.%d" % i)
                  for i in range(3)]
    for name in additional:
      open(name, "w").write("Some contents")
    filenames = list(set(filenames + additional))
    with self.test_session():
      star = tf.train.match_filenames_once(
          os.path.join(self.get_temp_dir(), "*"))
      question = tf.train.match_filenames_once(
          os.path.join(self.get_temp_dir(), "match_filenames.?"))
      one = tf.train.match_filenames_once(additional[1])
      tf.initialize_all_variables().run()
      self.assertItemsEqual(map(tf.compat.as_bytes, filenames), star.eval())
      self.assertItemsEqual(map(tf.compat.as_bytes, additional),
                            question.eval())
      self.assertItemsEqual([tf.compat.as_bytes(additional[1])], one.eval())


class LimitEpochsTest(tf.test.TestCase):

  def testNoLimit(self):
    with self.test_session():
      seven = tf.constant(7)
      seven_forever = tf.train.limit_epochs(seven)
      tf.initialize_all_variables().run()
      for i in range(100):
        self.assertEqual(7, seven_forever.eval())

  def testLimit(self):
    with self.test_session():
      love_me = tf.constant("Love Me")
      love_me_two_times = tf.train.limit_epochs(love_me, num_epochs=2)
      tf.initialize_all_variables().run()
      self.assertEqual(b"Love Me", love_me_two_times.eval())
      self.assertEqual(b"Love Me", love_me_two_times.eval())
      with self.assertRaises(tf.errors.OutOfRangeError):
        love_me_two_times.eval()


class StringInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session():
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 3
      queue = tf.train.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(strings) * num_epochs)
      dequeue = queue.dequeue()
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = dequeue_many.eval()
      self.assertAllEqual(strings * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session():
      strings = [b"a", b"b", b"c"]
      num_epochs = 600
      queue = tf.train.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=True, seed=271828)
      dequeue_many = queue.dequeue_many(len(strings))
      dequeue = queue.dequeue()
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the strings within an epoch and
      # count how often each possible order appears.
      expected = [b"abc", b"acb", b"bac", b"bca", b"cab", b"cba"]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = dequeue_many.eval()
        key = b"".join(output)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testNullStringPython(self):
    # Graph-construction time check for empty string list:
    with self.test_session():
      with self.assertRaises(ValueError):
        _ = tf.train.string_input_producer([])

  def testNullString(self):
    # Runtime check for empty string list.  This is slightly oblique:
    # The queue runner should die with an assertion error on the null
    # input tensor, causing the dequeue to fail with an OutOfRangeError.
    with self.test_session():
      coord = tf.train.Coordinator()
      queue = tf.train.string_input_producer(tf.constant([], dtype=tf.string))
      dequeue = queue.dequeue()
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners(coord=coord)
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      coord.request_stop()
      for thread in threads:
        thread.join()


class RangeInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session():
      num_epochs = 3
      range_size = 5
      queue = tf.train.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(range_size * num_epochs)
      dequeue = queue.dequeue()
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = dequeue_many.eval()
      self.assertAllEqual(list(xrange(range_size)) * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session():
      num_epochs = 200
      range_size = 2
      queue = tf.train.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=True, seed=314159)
      dequeue_many = queue.dequeue_many(range_size)
      dequeue = queue.dequeue()
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [12, 21]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = dequeue_many.eval()
        key = 10 * (output[0] + 1) + (output[1] + 1)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()


class SliceInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session() as sess:
      num_epochs = 3
      source_strings = [b"Alpha", b"Beta", b"Delta", b"Gamma"]
      source_ints = [2, 3, 5, 7]
      slices = tf.train.slice_input_producer(
          [source_strings, source_ints], num_epochs=num_epochs, shuffle=False)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      num_items = len(source_strings) * num_epochs
      output = [sess.run(slices) for _ in range(num_items)]
      out_strings, out_ints = zip(*output)
      self.assertAllEqual(source_strings * num_epochs, out_strings)
      self.assertAllEqual(source_ints * num_epochs, out_ints)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(slices)
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session() as sess:
      num_epochs = 1200
      source_strings = ["A", "B", "D", "G"]
      source_ints = [7, 3, 5, 2]
      slices = tf.train.slice_input_producer(
          [source_strings, source_ints], num_epochs=num_epochs, shuffle=True,
          seed=161803)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [b",".join(x) for x in
                  itertools.permutations([b"A7", b"B3", b"D5", b"G2"])]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = [sess.run(slices) for _ in range(len(source_strings))]
        key = b",".join([s + tf.compat.as_bytes(str(i)) for s, i in output])
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(slices)
      for thread in threads:
        thread.join()


class BatchTest(tf.test.TestCase):

  def testOneThread(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.batch([counter, "string"], batch_size=batch_size)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertAllEqual(results[1], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testOneThreadEnqueueMany(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      pre_batched = tf.train.batch([counter, "string"], batch_size=2)
      batched = tf.train.batch(pre_batched, enqueue_many=True,
                               batch_size=batch_size)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertAllEqual(results[1], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.batch([counter, "string"], batch_size=batch_size,
                               num_threads=4)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(results[1], [b"string"] * batch_size)
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()


class BatchJoinTest(tf.test.TestCase):

  def testTwoThreads(self):
    with self.test_session() as sess:
      # Two threads, the first generates (0..69, "a").
      num_a = 70
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, "b") 90 times and then stops.
      num_b = 90
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = tf.train.batch_join([[counter, "a"], [ninety_nine, "b"]],
                                    batch_size=batch_size)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        which_a = [i for i, s in enumerate(results[1]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if len(which_a) > 0 and len(which_b) > 0: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()


class ShuffleBatchTest(tf.test.TestCase):

  def testOneThread(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.shuffle_batch(
          [counter, "string"], batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=141421)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(results[1], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.shuffle_batch(
          [counter, "string"], batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=173205, num_threads=4)
      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(results[1], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()


class ShuffleBatchJoinTest(tf.test.TestCase):

  def testTwoThreads(self):
    with self.test_session() as sess:
      # Two threads, the first generates (0..24, "a").
      num_a = 25
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, "b") 35 times and then stops.
      num_b = 35
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = tf.train.shuffle_batch_join(
          [[counter, "a"], [ninety_nine, "b"]], batch_size=batch_size,
          capacity=32, min_after_dequeue=16, seed=223607)

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        which_a = [i for i, s in enumerate(results[1]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if len(which_a) > 0 and len(which_b) > 0: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Saw all the items from "a", but scrambled.
      self.assertItemsEqual(all_a, range(num_a))
      deltas = [all_a[i + 1] - all_a[i]
                for i in range(len(all_a) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()


if __name__ == "__main__":
  tf.test.main()
