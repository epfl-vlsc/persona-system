TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.bias_` {#TensorFlowRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowRegressor.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRegressor.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_params(deep=True)` {#TensorFlowRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor(name)` {#TensorFlowRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor_value(name)` {#TensorFlowRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_variable_names()` {#TensorFlowRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_variable_value(name)` {#TensorFlowRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.model_dir` {#TensorFlowRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.partial_fit(x, y)` {#TensorFlowRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class label in classification, real numbers in regression).

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.predict_proba(x, batch_size=None)` {#TensorFlowRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.restore(cls, path, config=None)` {#TensorFlowRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.save(path)` {#TensorFlowRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.set_params(**params)` {#TensorFlowRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Args:


*  <b>`**params`</b>: Parameters.

##### Returns:

  self

##### Raises:


*  <b>`ValueError`</b>: If params contain invalid names.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.train(input_fn, steps, monitors=None)` {#TensorFlowRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.weights_` {#TensorFlowRegressor.weights_}

Returns weights of the linear regression.


