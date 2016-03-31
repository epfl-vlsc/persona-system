# Distributed TensorFlow

This document shows how to create a cluster of TensorFlow servers, and how to
distribute a computation graph across that cluster. We assume that you are
familiar with the [basic concepts](../../get_started/basic_usage.md) of
writing TensorFlow programs.

## Quick start

The gRPC server is included as part of the nightly PIP packages, which you can
download from [the continuous integration
site](http://ci.tensorflow.org/view/Nightly/). Alternatively, you can build an
up-to-date PIP package by following [these installation instructions]
(https://www.tensorflow.org/versions/master/get_started/os_setup.html#create-the-pip-package-and-install).

Once you have successfully built the distributed TensorFlow components, you can
test your installation by starting a local server as follows:

```shell
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.GrpcServer.create_local_server()
>>> sess = tf.Session(server.target)
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

## Cluster definition

The `tf.GrpcServer.create_local_server()` method creates a single-process cluster.
To create a more realistic distributed cluster, you create a `tf.GrpcServer` by
passing in a `tf.ServerDef` that defines the membership of a TensorFlow cluster,
and then run multiple processes that each have the same cluster definition.

A `tf.ServerDef` comprises a cluster definition (`tf.ClusterDef`), which is the
same for all servers in a cluster; and a job name and task index that are unique
to a particular cluster.

For constructing a `tf.ClusterDef`, the `tf.make_cluster_def()` function enables you to specify the jobs and tasks as a Python dictionary, mapping job names to lists of network addresses. For example:

<table>
  <tr><th><code>tf.ClusterDef</code> construction</th><th>Available tasks</th>
  <tr>
    <td><code>tf.make_cluster_def({"local": ["localhost:2222", "localhost:2223"]})</code></td><td><code>/job:local/task:0<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><code>tf.make_cluster_def({<br/>&nbsp;&nbsp;&nbsp;&nbsp;"worker": ["worker0:2222", "worker1:2222", "worker2:2222"],<br/>&nbsp;&nbsp;&nbsp;&nbsp;"ps": ["ps0:2222", "ps1:2222"]})</code></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

The `server_def.job_name` and `server_def.task_index` fields select one of the
defined tasks from the `tf.ClusterDef`. For example, running the following code
in two different processes:

```python
# In task 0:
server_def = tf.ServerDef(
    cluster=tf.make_cluster_def({
        "local": ["localhost:2222", "localhost:2223"]}),
    job_name="local", task_index=0)
server = tf.GrpcServer(server_def)
```
```python
# In task 1:
server_def = tf.ServerDef(
    cluster=tf.make_cluster_def({
        "local": ["localhost:2222", "localhost:2223"]}),
    job_name="local", task_index=1)
server = tf.GrpcServer(server_def)
```

&hellip;will define and instantiate servers running on `localhost:2222` and
`localhost:2223`.

**N.B.** Manually specifying these cluster specifications can be tedious,
especially for large clusters. We are working on tools for launching tasks
programmatically, e.g. using a cluster manager like
[Kubernetes](http://kubernetes.io). If there are particular cluster managers for
which you'd like to see support, please raise a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues).

## Specifying distributed devices in your model

To place operations on a particular process, you can use the same
[`tf.device()`](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#device)
function that is used to specify whether ops run on the CPU or GPU. For example:

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

In the above example, the variables are created on two tasks in the `ps` job,
and the compute-intensive part of the model is created in the `worker`
job. TensorFlow will insert the appropriate data transfers between the jobs
(from `ps` to `worker` for the forward pass, and from `worker` to `ps` for
applying gradients).

## Replicated training

A common training configuration ("data parallel training") involves multiple
tasks in a `worker` job training the same model, using shared parameters hosted
in a one or more tasks in a `ps` job. Each task will typically run on a
different machine. There are many ways to specify this structure in TensorFlow,
and we are building libraries that will simplify the work of specifying a
replicated model. Possible approaches include:

* Building a single graph containing one set of parameters (in `tf.Variable`
  nodes pinned to `/job:ps`), and multiple copies of the "model" pinned to
  different tasks in `/job:worker`. Each copy of the model can have a different
  `train_op`, and one or more client threads can call `sess.run(train_ops[i])`
  for each worker `i`. This implements *asynchronous* training.

  This approach uses a single `tf.Session` whose target is one of the workers in
  the cluster.

* As above, but where the gradients from all workers are averaged. See the
  [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)
  for an example of this form of replication. This implements *synchronous*
  training.

* The "distributed trainer" approach uses multiple graphs&mdash;one per
  worker&mdash;where each graph contains one set of parameters (pinned to
  `/job:ps`) and one copy of the model (pinned to a particular
  `/job:worker/task:i`). The "container" mechanism is used to share variables
  between different graphs: when each variable is constructed, the optional
  `container` argument is specified with the same value in each copy of the
  graph. For large models, this can be more efficient, because the overall graph
  is smaller.

  This approach uses multiple `tf.Session` objects: one per worker process,
  where the `target` of each is the address of a different worker. The
  `tf.Session` objects can all be created in a single Python client, or you can
  use multiple Python clients to better distribute the trainer load.

## Glossary

<dl>
  <dt>Client</dt>
  <dd>
    A client is typically a program that builds a TensorFlow graph and
    constructs a `tensorflow::Session` to interact with a cluster. Clients are
    typically written in Python or C++. A single client process can directly
    interact with multiple TensorFlow servers (see "Replicated training" above),
    and a single server can serve multiple clients.
  </dd>
  <dt>Cluster</dt>
  <dd>
    A TensorFlow cluster comprises one or more TensorFlow servers, divided into
    a set of named jobs, which in turn comprise lists of tasks. A cluster is
    typically dedicated to a particular high-level objective, such as training a
    neural network, using many machines in parallel.
  </dd>
  <dt>Job</dt>
  <dd>
    A job comprises a list of "tasks", which typically serve a common
    purpose. For example, a job named `ps` (for "parameter server") typically
    hosts nodes that store and update variables; while a job named `worker`
    typically hosts stateless nodes that perform compute-intensive tasks.
    The tasks in a job typically run on different machines.
  </dd>
  <dt>Master service</dt>
  <dd>
    An RPC service that provides remote access to a set of distributed
    devices. The master service implements the <code>tensorflow::Session</code>
    interface, and is responsible for coordinating work across one or more
    "worker services".
  </dd>
  <dt>Task</dt>
  <dd>
    A task typically corresponds to a single TensorFlow server process,
    belonging to a particular "job" and with a particular index within that
    job's list of tasks.
  </dd>
  <dt>TensorFlow server</dt>
  <dd>
    A process running a <code>tf.GrpcServer</code> instance, which is a
    member of a cluster, and exports a "master service" and "worker service".
  </dd>
  <dt>Worker service</dt>
  <dd>
    An RPC service that executes parts of a TensorFlow graph using its local
    devices. A worker service implements <a href=
    "https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto"
    ><code>worker_service.proto</code></a>.
  </dd>
</dl>
