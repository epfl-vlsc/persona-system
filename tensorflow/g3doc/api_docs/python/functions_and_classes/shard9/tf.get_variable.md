### `tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True)` {#get_variable}

Gets an existing variable with these parameters or create a new one.

This function prefixes the name with the current variable scope
and performs reuse checks. See the
[Variable Scope How To](../../how_tos/variable_scope/index.md)
for an extensive description of how reusing works. Here is a basic example:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True)
    v1 = tf.get_variable("v")  # The same as v above.
```

If initializer is `None` (the default), the default initializer passed in
the variable scope will be used. If that one is `None` too, a
`UniformUnitScalingInitializer` will be used. The initializer can also be
a Tensor, in which case the variable is initialized to this value and shape.

Similarly, if the regularizer is `None` (the default), the default regularizer
passed in the variable scope will be used (if that is `None` too,
then by default no regularization is performed).

If a partitioner is provided, first a sharded `Variable` is created
via `_get_partitioned_variable_list`, and the return value is a
`Tensor` composed of the shards concatenated along the partition axis.

Some useful partitioners are available.  See, e.g.,
`variable_axis_size_partitioner`.

##### Args:


*  <b>`name`</b>: The name of the new or existing variable.
*  <b>`shape`</b>: Shape of the new or existing variable.
*  <b>`dtype`</b>: Type of the new or existing variable (defaults to `DT_FLOAT`).
*  <b>`initializer`</b>: Initializer for the variable if one is created.
*  <b>`regularizer`</b>: A (Tensor -> Tensor or None) function; the result of
    applying it on a newly created variable will be added to the collection
    GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
*  <b>`trainable`</b>: If `True` also add the variable to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`collections`</b>: List of graph collections keys to add the Variable to.
    Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
    If partitioning is enabled and used, the concatenated return value
    is also added to collection `GraphKeys.CONCATENATED_VARIABLES`.
*  <b>`caching_device`</b>: Optional device string or function describing where the
    Variable should be cached for reading.  Defaults to the Variable's
    device.  If not `None`, caches on another device.  Typical use is to
    cache on the device where the Ops using the Variable reside, to
    deduplicate copying through `Switch` and other conditional statements.
*  <b>`partitioner`</b>: Optional callable that accepts a fully defined `TensorShape`
    and `dtype` of the Variable to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
*  <b>`validate_shape`</b>: If False, allows the variable to be initialized with a
      value of unknown shape. If True, the default, the shape of initial_value
      must be known.

##### Returns:

  The created or existing variable.

##### Raises:


*  <b>`ValueError`</b>: when creating a new variable and shape is not declared,
    or when violating reuse during variable creation. Reuse is set inside
    `variable_scope`.

