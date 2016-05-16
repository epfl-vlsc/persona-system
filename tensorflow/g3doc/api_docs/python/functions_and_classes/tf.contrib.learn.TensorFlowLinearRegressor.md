TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowLinearRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.bias_` {#TensorFlowLinearRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.fit(X, y, monitor=None, logdir=None)` {#TensorFlowLinearRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`monitor`</b>: Monitor object to print training progress and invoke early
    stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_params(deep=True)` {#TensorFlowLinearRegressor.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor(name)` {#TensorFlowLinearRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor_value(name)` {#TensorFlowLinearRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_variable_names()` {#TensorFlowLinearRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.partial_fit(X, y)` {#TensorFlowLinearRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class label in classification, real numbers in regression).

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict(X, axis=1, batch_size=None)` {#TensorFlowLinearRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
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

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict_proba(X, batch_size=None)` {#TensorFlowLinearRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.restore(cls, path, config=None)` {#TensorFlowLinearRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.save(path)` {#TensorFlowLinearRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.score(X, y, sample_weight=None)` {#TensorFlowLinearRegressor.score}

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the regression
sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.set_params(**params)` {#TensorFlowLinearRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.weights_` {#TensorFlowLinearRegressor.weights_}

Returns weights of the linear regression.


