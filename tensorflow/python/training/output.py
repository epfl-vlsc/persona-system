
# Stuart Byma
# Basically copied from input.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import queue_runner

def _validate_join(tensor_list_list):
  tensor_list_list = [ops.convert_n_to_tensor_or_indexed_slices(tl)
                      for tl in tensor_list_list]
  if not tensor_list_list:
    raise ValueError("Expected at least one input in batch_join().")
  return tensor_list_list

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
  if shapes is None:
    l = len(tensor_list_list[0])
    shapes = [_merge_shapes(
        [tl[i].get_shape().as_list() for tl in tensor_list_list], enqueue_many)
              for i in xrange(l)]
  return shapes

def _enqueue_join(queue, tensor_list_list, enqueue_many):
  if enqueue_many:
    enqueue_ops = [queue.enqueue_many(tl) for tl in tensor_list_list]
  else:
    enqueue_ops = [queue.enqueue(tl) for tl in tensor_list_list]
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def unbatch_join(tensor_list_list, capacity=32, enqueue_many=False,
               shapes=None, shared_name=None, name=None):
  """Runs a list of tensors to fill a queue, outputs single examples for
  writing to file.

  Enqueues a different list of tensors in different threads.
  Implemented using a queue -- a `QueueRunner` for the queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  `len(tensor_list_list)` threads will be started,
  with thread `i` enqueuing the tensors from
  `tensor_list_list[i]`. `tensor_list_list[i1][j]` must match
  `tensor_list_list[i2][j]` in type and shape, except in the first
  dimension if `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensor_list_list[i]` is assumed
  to represent a single example. An input tensor `x` will be output 
  unchanged.

  If `enqueue_many` is `True`, `tensor_list_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensor_list_list[i]` should have the
  same size in the first dimension.  The slices of any input tensor
  `x` are treated as examples, and the output tensors will have shape
  `x.shape[1:]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* You must ensure that either (i) the `shapes` argument is
  passed, or (ii) all of the tensors in `tensor_list_list` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  Args:
    tensor_list_list: A list of tuples of tensors to enqueue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
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
  with ops.op_scope(_flatten(tensor_list_list), name, "batch_join") as name:
    tensor_list_list = _validate_join(tensor_list_list)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = data_flow_ops.FIFOQueue(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))
    return queue.dequeue_many(1, name=name)

