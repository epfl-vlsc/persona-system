Base class for queue implementations.

A queue is a TensorFlow data structure that stores tensors across
multiple steps, and exposes operations that enqueue and dequeue
tensors.

Each queue element is a tuple of one or more tensors, where each
tuple component has a static dtype, and may have a static shape. The
queue implementations support versions of enqueue and dequeue that
handle single elements, versions that support enqueuing and
dequeuing a batch of elements at once.

See [`tf.FIFOQueue`](#FIFOQueue) and
[`tf.RandomShuffleQueue`](#RandomShuffleQueue) for concrete
implementations of this class, and instructions on how to create
them.

- - -

#### `tf.QueueBase.enqueue(vals, name=None)` {#QueueBase.enqueue}

Enqueues one element to this queue.

If the queue is full when this operation executes, it will block
until the element has been enqueued.

##### Args:


*  <b>`vals`</b>: The tuple of `Tensor` objects to be enqueued.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a new tuple of tensors to the queue.


- - -

#### `tf.QueueBase.enqueue_many(vals, name=None)` {#QueueBase.enqueue_many}

Enqueues zero or more elements to this queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tensors in `vals` must have the
same size in the 0th dimension.

If the queue is full when this operation executes, it will block
until all of the elements have been enqueued.

##### Args:


*  <b>`vals`</b>: The tensor or tuple of tensors from which the queue elements
    are taken.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a batch of tuples of tensors to the queue.



- - -

#### `tf.QueueBase.dequeue(name=None)` {#QueueBase.dequeue}

Dequeues one element from this queue.

If the queue is empty when this operation executes, it will block
until there is an element to dequeue.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of tensors that was dequeued.


- - -

#### `tf.QueueBase.dequeue_many(n, name=None)` {#QueueBase.dequeue_many}

Dequeues and concatenates `n` elements from this queue.

This operation concatenates queue-element component tensors along
the 0th dimension to make a single component tensor.  All of the
components in the dequeued tuple will have size `n` in the 0th dimension.

If the queue is closed and there are less than `n` elements left, then an
`OutOfRange` exception is raised.

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.



- - -

#### `tf.QueueBase.size(name=None)` {#QueueBase.size}

Compute the number of elements in this queue.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar tensor containing the number of elements in this queue.



- - -

#### `tf.QueueBase.close(cancel_pending_enqueues=False, name=None)` {#QueueBase.close}

Closes this queue.

This operation signals that no more elements will be enqueued in
the given queue. Subsequent `enqueue` and `enqueue_many`
operations will fail. Subsequent `dequeue` and `dequeue_many`
operations will continue to succeed if sufficient elements remain
in the queue. Subsequent `dequeue` and `dequeue_many` operations
that would block will fail immediately.

If `cancel_pending_enqueues` is `True`, all pending requests will also
be cancelled.

##### Args:


*  <b>`cancel_pending_enqueues`</b>: (Optional.) A boolean, defaulting to
    `False` (described above).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that closes the queue.



#### Other Methods
- - -

#### `tf.QueueBase.__init__(dtypes, shapes, queue_ref)` {#QueueBase.__init__}

Constructs a queue object from a queue reference.

##### Args:


*  <b>`dtypes`</b>: A list of types.  The length of dtypes must equal the number
    of tensors in each element.
*  <b>`shapes`</b>: Constraints on the shapes of tensors in an element:
    A list of shape tuples or None. This list is the same length
    as dtypes.  If the shape of any tensors in the element are constrained,
    all must be; shapes can be None if the shapes should not be constrained.
*  <b>`queue_ref`</b>: The queue reference, i.e. the output of the queue op.


- - -

#### `tf.QueueBase.dequeue_up_to(n, name=None)` {#QueueBase.dequeue_up_to}

Dequeues and concatenates `n` elements from this queue.

**Note** This operation is not supported by all queues.  If a queue does not
support DequeueUpTo, then an Unimplemented exception is raised.

This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size `n` in the 0th dimension.

If the queue is closed and there are more than `0` but less than `n`
elements remaining, then instead of raising an `OutOfRange` exception like
`dequeue_many`, the remaining elements are returned immediately.
If the queue is closed and there are `0` elements left in the queue, then
an `OutOfRange` exception is raised just like in `dequeue_many`.
Otherwise the behavior is identical to `dequeue_many`:

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.


- - -

#### `tf.QueueBase.dtypes` {#QueueBase.dtypes}

The list of dtypes for each component of a queue element.


- - -

#### `tf.QueueBase.from_list(index, queues)` {#QueueBase.from_list}

Create a queue using the queue reference from `queues[index]`.

##### Args:


*  <b>`index`</b>: An integer scalar tensor that determines the input that gets
    selected.
*  <b>`queues`</b>: A list of `QueueBase` objects.

##### Returns:

  A `QueueBase` object.

##### Raises:


*  <b>`TypeError`</b>: When `queues` is not a list of `QueueBase` objects,
    or when the data types of `queues` are not all the same.


- - -

#### `tf.QueueBase.name` {#QueueBase.name}

The name of the underlying queue.


- - -

#### `tf.QueueBase.queue_ref` {#QueueBase.queue_ref}

The underlying queue reference.


