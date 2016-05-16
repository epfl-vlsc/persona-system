<!-- This file is machine generated: DO NOT EDIT! -->

# Asserts and boolean checks.
[TOC]

## Asserts and Boolean Checks

- - -

### `tf.assert_negative(x, data=None, summarize=None, name=None)` {#assert_negative}

Assert the condition `x < 0` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_negative(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_negative(x)], x)
```

Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
If `x` is empty this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_negative".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` is all negative.


- - -

### `tf.assert_positive(x, data=None, summarize=None, name=None)` {#assert_positive}

Assert the condition `x > 0` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_positive(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_positive(x)], x)
```

Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
If `x` is empty this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_positive".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` is all positive.


- - -

### `tf.assert_non_negative(x, data=None, summarize=None, name=None)` {#assert_non_negative}

Assert the condition `x >= 0` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_non_negative(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_non_negative(x)], x)
```

Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
If `x` is empty this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "assert_non_negative".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` is all non-negative.


- - -

### `tf.assert_non_positive(x, data=None, summarize=None, name=None)` {#assert_non_positive}

Assert the condition `x <= 0` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_non_positive(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_non_positive(x)], x)
```

Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
If `x` is empty this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "assert_non_positive".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` is all non-positive.


- - -

### `tf.assert_equal(x, y, data=None, summarize=None, name=None)` {#assert_equal}

Assert the condition `x == y` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_equal(x, y)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_equal(x, y)], x)
```

This condition holds if for every pair of (possibly broadcast) elements
`x[i]`, `y[i]`, we have `x[i] == y[i]`.
If both `x` and `y` are empty, this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`y`</b>: Numeric `Tensor`, same dtype as and broadcastable to `x`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`, `y`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_equal".

##### Returns:

  Op that raises `InvalidArgumentError` if `x == y` is False.


- - -

### `tf.assert_less(x, y, data=None, summarize=None, name=None)` {#assert_less}

Assert the condition `x < y` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_less(x, y)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_less(x, y)], x)
```

This condition holds if for every pair of (possibly broadcast) elements
`x[i]`, `y[i]`, we have `x[i] < y[i]`.
If both `x` and `y` are empty, this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`y`</b>: Numeric `Tensor`, same dtype as and broadcastable to `x`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`, `y`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_less".

##### Returns:

  Op that raises `InvalidArgumentError` if `x < y` is False.


- - -

### `tf.assert_less_equal(x, y, data=None, summarize=None, name=None)` {#assert_less_equal}

Assert the condition `x <= y` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_less_equal(x, y)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_less_equal(x, y)], x)
```

This condition holds if for every pair of (possibly broadcast) elements
`x[i]`, `y[i]`, we have `x[i] <= y[i]`.
If both `x` and `y` are empty, this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`y`</b>: Numeric `Tensor`, same dtype as and broadcastable to `x`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`, `y`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_less_equal"

##### Returns:

  Op that raises `InvalidArgumentError` if `x <= y` is False.


- - -

### `tf.assert_rank(x, rank, data=None, summarize=None, name=None)` {#assert_rank}

Assert `x` has rank equal to `rank`.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_rank(x, 2)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_rank(x, 2)], x)
```

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`rank`</b>: Scalar `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_rank".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` has specified rank.

##### Raises:


*  <b>`ValueError`</b>: If static checks determine `x` has wrong rank.


- - -

### `tf.assert_rank_at_least(x, rank, data=None, summarize=None, name=None)` {#assert_rank_at_least}

Assert `x` has rank equal to `rank` or higher.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_rank_at_least(x, 2)], x)
```

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`rank`</b>: Scalar `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "assert_rank_at_least".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` has specified rank or higher.

##### Raises:


*  <b>`ValueError`</b>: If static checks determine `x` has wrong rank.


- - -

### `tf.assert_integer(x, data=None, summarize=None, name=None)` {#assert_integer}

Assert that `x` is of integer dtype.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_integer(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_integer(x)], x)
```

##### Args:


*  <b>`x`</b>: `Tensor` whose basetype is integer and is not quantized.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_integer".

##### Returns:

  Op that raises `InvalidArgumentError` if `x == y` is False.


- - -

### `tf.is_non_decreasing(x, name=None)` {#is_non_decreasing}

Returns `True` if `x` is non-decreasing.

Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
If `x` has less than two elements, it is trivially non-decreasing.

See also:  `is_strictly_increasing`

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "is_non_decreasing"

##### Returns:

  Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

##### Raises:


*  <b>`TypeError`</b>: if `x` is not a numeric tensor.


- - -

### `tf.is_numeric_tensor(tensor)` {#is_numeric_tensor}




- - -

### `tf.is_strictly_increasing(x, name=None)` {#is_strictly_increasing}

Returns `True` if `x` is strictly increasing.

Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
If `x` has less than two elements, it is trivially strictly increasing.

See also:  `is_non_decreasing`

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "is_strictly_increasing"

##### Returns:

  Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

##### Raises:


*  <b>`TypeError`</b>: if `x` is not a numeric tensor.


