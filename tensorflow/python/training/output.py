
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

def _serialize_sparse_tensors(tensor_list, enqueue_many):
  """Serialize SparseTensors for feeding into batch, etc."""
  sparse_info_list = [
      _SparseMetaData(sparse=True,
                      dtype=t.dtype,
                      rank=t.shape.get_shape().with_rank(1)[0])
      if isinstance(t, ops.SparseTensor)
      else _SparseMetaData(False, None, None)
      for t in tensor_list]

  def _maybe_serialize(t, sparse):
    if not sparse:
      return t
    return (sparse_ops.serialize_many_sparse(t) if enqueue_many
            else sparse_ops.serialize_sparse(t))

  serialized_list = [
      _maybe_serialize(t, info.sparse) for (t, info)
      in zip(tensor_list, sparse_info_list)]

  return serialized_list, sparse_info_list

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

def _enqueue(queue, tensor_list, threads, enqueue_many):
  if enqueue_many:
    enqueue_ops = [queue.enqueue_many(tensor_list)] * threads
  else:
    enqueue_ops = [queue.enqueue(tensor_list)] * threads
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))

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
    tensor_list = _validate(tensor_list)
    (tensor_list, sparse_info) = _serialize_sparse_tensors(
        tensor_list, enqueue_many)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    dequeued = queue.dequeue(name=name)  #dequeue single tensors
    dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued

