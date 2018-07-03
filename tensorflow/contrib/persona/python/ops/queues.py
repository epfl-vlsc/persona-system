

# custom queue constructs based on those from TF
# duplicates some code :-/


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python import training
from tensorflow.python.training import queue_runner
from tensorflow.python.summary import summary

from six.moves import xrange  # pylint: disable=redefined-builtin

import json
import os

def _validate(tensor_list):
  tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
  if not tensor_list:
    raise ValueError("Expected at least one tensor in batch().")
  return tensor_list

def _validate_join(tensor_list_list):
  tensor_list_list = [ops.convert_n_to_tensor_or_indexed_slices(tl)
                      for tl in tensor_list_list]
  if not tensor_list_list:
    raise ValueError("Expected at least one input in batch_join().")
  return tensor_list_list

def _flatten(tensor_list_list):
  return [tensor for tensor_list in tensor_list_list for tensor in tensor_list]

def _dtypes(tensor_list_list):
  all_types = [[t.dtype for t in tl] for tl in tensor_list_list]
  types = all_types[0]
  for other_types in all_types[1:]:
    if other_types != types:
      raise TypeError("Expected types to be consistent: %s vs. %s." %
                      (", ".join(x.name for x in types),
                       ", ".join(x.name for x in other_types)))
  return types

def _shapes(tensor_list_list, shapes, enqueue_many):
  """Calculate and merge the shapes of incoming tensors.

  Args:
    tensor_list_list: List of tensor lists.
    shapes: List of shape tuples corresponding to tensors within the lists.
    enqueue_many: Boolean describing whether shapes will be enqueued as
      batches or individual entries.

  Returns:
    A list of shapes aggregating shape inference info from `tensor_list_list`,
    or returning `shapes` if it is not `None`.

  Raises:
    ValueError: If any of the inferred shapes in `tensor_list_list` lack a
      well defined rank.
  """
  if shapes is None:
    len0 = len(tensor_list_list[0])

    for tl in tensor_list_list:
      for i in xrange(len0):
        if tl[i].get_shape().ndims is None:
          raise ValueError("Cannot infer Tensor's rank: %s" % tl[i])

    shapes = [_merge_shapes(
        [tl[i].get_shape().as_list() for tl in tensor_list_list], enqueue_many)
              for i in xrange(len0)]
  return shapes

def _merge_shapes(shape_list, enqueue_many):
  shape_list = [tensor_shape.as_shape(s) for s in shape_list]
  if enqueue_many:
    # We want the shapes without the leading batch dimension.
    shape_list = [s.with_rank_at_least(1)[1:] for s in shape_list]
  merged_shape = shape_list[0]
  for s in shape_list[1:]:
    merged_shape.merge_with(s)
  return merged_shape.as_list()


def _enqueue_join(queue, tensor_list_list, enqueue_many, keep_input):
  """Enqueue `tensor_list_list` in `queue`."""
  if enqueue_many:
    enqueue_fn = queue.enqueue_many
  else:
    enqueue_fn = queue.enqueue
  if keep_input is None:
    enqueue_ops = [enqueue_fn(tl) for tl in tensor_list_list]
  else:
    enqueue_ops = [control_flow_ops.cond(
        keep_input,
        lambda: enqueue_fn(tl),
        control_flow_ops.no_op) for tl in tensor_list_list]
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _enqueue(queue, tensor_list, threads, enqueue_many, keep_input):
  """Enqueue `tensor_list` in `queue`."""
  if enqueue_many:
    enqueue_fn = queue.enqueue_many
  else:
    enqueue_fn = queue.enqueue
  if keep_input is None:
    enqueue_ops = [enqueue_fn(tensor_list)] * threads
  else:
    enqueue_ops = [control_flow_ops.cond(
        keep_input,
        lambda: enqueue_fn(tensor_list),
        control_flow_ops.no_op)] * threads
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _which_queue(dynamic_pad):
  return (data_flow_ops.PaddingFIFOQueue if dynamic_pad
          else data_flow_ops.FIFOQueue)

def batch_pdq(tensor_list, batch_size, num_threads=1, capacity=32, num_dq_ops=1,
          enqueue_many=False, shapes=None, dynamic_pad=False,
          shared_name=None, name=None):
  """Creates batches of tensors in `tensor_list`. Can dequeue multiple
  batches at a .

  This function is implemented using a queue. A `QueueRunner` for the
  queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  If `enqueue_many` is `False`, `tensor_list` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensor_list` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensor_list` should have the same size in the
  first dimension.  If an input tensor has shape `[*, x, y, z]`, the
  output will have shape `[batch_size, x, y, z]`.  The `capacity` argument
  controls the how long the prefetching is allowed to grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensor_list` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have shape `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  Args:
    tensor_list: The list of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensor_list`.
    num_dq_ops: num dq ops
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as `tensor_list`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list`.
  """
  with ops.name_scope(name, "batch_pdq", tensor_list) as name:
    tensor_list = _validate(tensor_list)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many, None)
    summary.scalar(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if batch_size == 1:
      dequeued = [queue.dequeue(name=name+"_%d"%i) for i in range(num_dq_ops)]
    else:
      dequeued = [queue.dequeue_many(batch_size, name=name+"_%d"%i) for i in range(num_dq_ops)]
    return dequeued

def batch_join_pdq(tensor_list_list, batch_size, num_dq_ops=1, capacity=32,
               enqueue_many=False,shapes=None, dynamic_pad=False,
               shared_name=None, name=None):
  """Runs a list of tensors to fill a queue to create batches of examples.
  This is a parallel dequeue batch join, and returns `num_dq_ops *
  dequeue_many` ops.

  Enqueues a different list of tensors in different threads.
  Implemented using a queue -- a `QueueRunner` for the queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  `len(tensor_list_list)` threads will be started,
  with thread `i` enqueuing the tensors from
  `tensor_list_list[i]`. `tensor_list_list[i1][j]` must match
  `tensor_list_list[i2][j]` in type and shape, except in the first
  dimension if `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensor_list_list[i]` is assumed
  to represent a single example. An input tensor `x` will be output as a
  tensor with shape `[batch_size] + x.shape`.

  If `enqueue_many` is `True`, `tensor_list_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensor_list_list[i]` should have the
  same size in the first dimension.  The slices of any input tensor
  `x` are treated as examples, and the output tensors will have shape
  `[batch_size] + x.shape[1:]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensor_list` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have value `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  Args:
    tensor_list_list: A list of tuples of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as
    `tensor_list_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
  with ops.name_scope(name, "batch_join_pdq",_flatten(tensor_list_list)) as name:
    tensor_list_list = _validate_join(tensor_list_list)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many, None)
    summary.scalar(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if batch_size == 1:
      dequeued = [queue.dequeue(name=name+"_%d"%i) for i in range(num_dq_ops)]
    else:
      dequeued = [queue.dequeue_many(batch_size, name=name+"_%d"%i) for i in range(num_dq_ops)]
    return dequeued
