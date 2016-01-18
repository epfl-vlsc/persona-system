<!-- This file is machine generated: DO NOT EDIT! -->

# Control Flow

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](../../api_docs/python/framework.md#convert_to_tensor).

[TOC]

## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control
the execution of operations and add conditional dependencies to your graph.

- - -

### `tf.identity(input, name=None)` {#identity}

Return a tensor with the same shape and contents as the input tensor or value.

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.tuple(tensors, name=None, control_inputs=None)` {#tuple}

Group tensors together.

This creates a tuple of tensors with the same values as the `tensors`
argument, except that the value of each tensor is only returned after the
values of all tensors have been computed.

`control_inputs` contains additional ops that have to finish before this op
finishes, but whose outputs are not returned.

This can be used as a "join" mechanism for parallel computations: all the
argument tensors can be computed in parallel, but the values of any tensor
returned by `tuple` are only available after all the parallel computations
are done.

See also `group` and `with_dependencies`.

##### Args:


*  <b>`tensors`</b>: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
*  <b>`name`</b>: (optional) A name to use as a `name_scope` for the operation.
*  <b>`control_inputs`</b>: List of additional ops to finish before returning.

##### Returns:

  Same as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
*  <b>`TypeError`</b>: If `control_inputs` is not a list of `Operation` or `Tensor`
    objects.


- - -

### `tf.group(*inputs, **kwargs)` {#group}

Create an op that groups multiple operations.

When this op finishes, all ops in `input` have finished. This op has no
output.

See also `tuple` and `with_dependencies`.

##### Args:


*  <b>`*inputs`</b>: One or more tensors to group.
*  <b>`**kwargs`</b>: Optional parameters to pass when constructing the NodeDef.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  An Operation that executes all its inputs.

##### Raises:


*  <b>`ValueError`</b>: If an unknown keyword argument is provided, or if there are
              no inputs.


- - -

### `tf.no_op(name=None)` {#no_op}

Does nothing. Only useful as a placeholder for control edges.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

### `tf.count_up_to(ref, limit, name=None)` {#count_up_to}

Increments 'ref' until it reaches 'limit'.

This operation outputs "ref" after the update is done.  This makes it
easier to chain operations that need to use the updated value.

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Must be one of the following types: `int32`, `int64`.
    Should be from a scalar `Variable` node.
*  <b>`limit`</b>: An `int`.
    If incrementing ref would bring it above limit, instead generates an
    'OutOfRange' error.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `ref`.
  A copy of the input before increment. If nothing else modifies the
  input, the values produced will all be distinct.


- - -

### `tf.cond(pred, fn1, fn2, name=None)` {#cond}

Return either 'fn1()' or 'fn2()' based on the boolean predicate 'pred'.

`fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
the same number and type of outputs.

##### Args:


*  <b>`pred`</b>: A scalar determining whether to return the result of `fn1` or `fn2`.
*  <b>`fn1`</b>: The function to be performed if pred is true.
*  <b>`fn2`</b>: The function to be performed if pref is false.
*  <b>`name`</b>: Optional name prefix for the returned tensors.

##### Returns:

  Tensors returned by the call to either `fn1` or `fn2`. If the functions
  return a singleton list, the element is extracted from the list.

##### Raises:


*  <b>`TypeError`</b>: if `fn1` or `fn2` is not callable.
*  <b>`ValueError`</b>: if `fn1` and `fn2` do not return the same number of tensors, or
              return tensors of different types.


*  <b>`Example`</b>: 
```python
  x = constant(2)
  y = constant(5)
  def f1(): return constant(17)
  def f2(): return constant(23)
  r = cond(math_ops.less(x, y), f1, f2)
  # r is set to f1()
```



## Logical Operators

TensorFlow provides several operations that you can use to add logical operators
to your graph.

- - -

### `tf.logical_and(x, y, name=None)` {#logical_and}

Returns the truth value of x AND y element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_not(x, name=None)` {#logical_not}

Returns the truth value of NOT x element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_or(x, y, name=None)` {#logical_or}

Returns the truth value of x OR y element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_xor(x, y, name='LogicalXor')` {#logical_xor}

x ^ y = (x | y) & ~(x & y).



## Comparison Operators

TensorFlow provides several operations that you can use to add comparison
operators to your graph.

- - -

### `tf.equal(x, y, name=None)` {#equal}

Returns the truth value of (x == y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.not_equal(x, y, name=None)` {#not_equal}

Returns the truth value of (x != y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.less(x, y, name=None)` {#less}

Returns the truth value of (x < y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.less_equal(x, y, name=None)` {#less_equal}

Returns the truth value of (x <= y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.greater(x, y, name=None)` {#greater}

Returns the truth value of (x > y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.greater_equal(x, y, name=None)` {#greater_equal}

Returns the truth value of (x >= y) element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.select(condition, t, e, name=None)` {#select}

Selects elements from `t` or `e`, depending on `condition`.

The `condition`, `t`, and `e` tensors must all have the same shape,
and the output will also have that shape. The `condition` tensor acts
as an element-wise mask that chooses, based on the value at each
element, whether the corresponding element in the output should be
taken from `t` (if true) or `e` (if false). For example:

For example:

```prettyprint
# 'condition' tensor is [[True, False]
#                        [True, False]]
# 't' is [[1, 1],
#         [1, 1]]
# 'e' is [[2, 2],
#         [2, 2]]
select(condition, t, e) ==> [[1, 2],
                             [1, 2]]
```

##### Args:


*  <b>`condition`</b>: A `Tensor` of type `bool`.
*  <b>`t`</b>: A `Tensor` with the same shape as `condition`.
*  <b>`e`</b>: A `Tensor` with the same type and shape as `t`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with the same type and shape as `t` and `e`.


- - -

### `tf.where(input, name=None)` {#where}

Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`input`. Indices are output in row-major order.

For example:

```prettyprint
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

##### Args:


*  <b>`input`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.



## Debugging Operations

TensorFlow provides several operations that you can use to validate values and
debug your graph.

- - -

### `tf.is_finite(x, name=None)` {#is_finite}

Returns which elements of x are finite.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.is_inf(x, name=None)` {#is_inf}

Returns which elements of x are Inf.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.is_nan(x, name=None)` {#is_nan}

Returns which elements of x are NaN.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.verify_tensor_all_finite(t, msg, name=None)` {#verify_tensor_all_finite}

Assert that the tensor does not contain any NaN's or Inf's.

##### Args:


*  <b>`t`</b>: Tensor to check.
*  <b>`msg`</b>: Message to log on failure.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Same tensor as `t`.


- - -

### `tf.check_numerics(tensor, message, name=None)` {#check_numerics}

Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

##### Args:


*  <b>`tensor`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`message`</b>: A `string`. Prefix of the error message.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.


- - -

### `tf.add_check_numerics_ops()` {#add_check_numerics_ops}

Connect a `check_numerics` to every floating point tensor.

`check_numerics` operations themselves are added for each `float` or `double`
tensor in the graph. For all ops in the graph, the `check_numerics` op for
all of its (`float` or `double`) inputs is guaranteed to run before the
`check_numerics` op on any of its outputs.

##### Returns:

  A `group` op depending on all `check_numerics` ops added.


- - -

### `tf.Assert(condition, data, summarize=None, name=None)` {#Assert}

Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

##### Args:


*  <b>`condition`</b>: The condition to evaluate.
*  <b>`data`</b>: The tensors to print out when condition is false.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).


- - -

### `tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)` {#Print}

Prints a list of tensors.

This is an identity op with the side effect of printing `data` when
evaluating.

##### Args:


*  <b>`input_`</b>: A tensor passed through this op.
*  <b>`data`</b>: A list of tensors to print out when op is evaluated.
*  <b>`message`</b>: A string, prefix of the error message.
*  <b>`first_n`</b>: Only log `first_n` number of times. Negative numbers log always;
           this is the default.
*  <b>`summarize`</b>: Only print this many entries of each tensor. If None, then a
             maximum of 3 elements are printed per input tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same tensor as `input_`.


