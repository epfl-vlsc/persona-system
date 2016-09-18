
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
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import queue_runner
#reusing a bunch of the helper functions from input.py
from tensorflow.python.training import input


def unbatch(tensor_list, num_threads=1, capacity=32,
          enqueue_many=False, shapes=None, dynamic_pad=False,
          shared_name=None, name=None):
  """ Unbatches tensors in tensor list and returns single tensors, 
  generally for writing out to files.

  This function is implemented using a queue. A `QueueRunner` for the
  queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  If `enqueue_many` is `False`, `tensor_list` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensor_list` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensor_list` should have the same size in the
  first dimension.  If an input tensor has shape `[*, x, y, z]`, the
  output will have shape `[x, y, z]`.  The `capacity` argument
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
    num_threads: The number of threads enqueuing `tensor_list`.
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
    A single tensor with the same number and types as `tensor_list`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list`.
  """
  with ops.op_scope(tensor_list, name, "unbatch") as name:
    tensor_list = input._validate(tensor_list)
    (tensor_list, sparse_info) = input._serialize_sparse_tensors(
        tensor_list, enqueue_many)
    types = input._dtypes([tensor_list])
    shapes = input._shapes([tensor_list], shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = input._which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    input._enqueue(queue, tensor_list, num_threads, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    dequeued = queue.dequeue(name=name)  #dequeue single tensors
    dequeued = input._deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued

def unbatch_join(tensor_list_list, capacity=32, batch_size=1, enqueue_many=False,
               shapes=None, dynamic_pad=False,
               shared_name=None, name=None):
  """Runs a list of tensors to fill a queue to create batches of examples.

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
  with ops.op_scope(input._flatten(tensor_list_list), name, "unbatch_join") as name:
    tensor_list_list = input._validate_join(tensor_list_list)
    tensor_list_list, sparse_info = input._serialize_sparse_tensors_join(
        tensor_list_list, enqueue_many)
    types = input._dtypes(tensor_list_list)
    shapes = input._shapes(tensor_list_list, shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = input._which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    input._enqueue_join(queue, tensor_list_list, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = input._deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued

