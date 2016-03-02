# TensorBoard

TensorBoard is a suite of web applications for inspecting and understanding your
TensorFlow runs and graphs.


### Basic Usage

Before running TensorBoard, make sure you have
generated summary data in a log directory by creating a `SummaryWriter`:

```python
# sess.graph_def is the graph definition.
summary_writer = tf.train.SummaryWriter('/path/to/logs', sess.graph_def)
```

For more details, see [this tutorial](http://www.tensorflow.org/how_tos/summaries_and_tensorboard/index.html#serializing-the-data).
Then run TensorBoard and provide the log directory:

```
python tensorflow/tensorboard/tensorboard.py --logdir=path/to/logs
# or if installed via pip, run:
tensorboard --logdir=path/to/logs

# if building from source
bazel build tensorflow/tensorboard:tensorboard
./bazel-bin/tensorflow/tensorboard/tensorboard --logdir=path/to/logs

# then connect to http://localhost:6006
```

Note that TensorBoard requires a `logdir` to read logs from. For info on
configuring TensorBoard, run `tensorboard --help`.

### Comparing Multiple Runs

TensorBoard can compare many "runs" of TensorFlow with each other. For example,
suppose you have two MNIST models with slightly different graphs or different
learning rates, but both share tags, e.g. `"cross_entropy"`, `"loss"`,
`"activations"`. TensorBoard has the capability to draw these charts on top of
each other, so you can easily see which model is outperforming.

To use this functionality, ensure that the summaries from the two runs are being
written to separate directories, for example:

```
./mnist_runs
./mnist_runs/run1/.*tfevents.*
./mnist_runs/run2/.*tfevents.*
```

Now, if you pass .../mnist_runs/run1 as the `logdir` to TensorBoard, you will
visualize training data from that first run. But, if you instead pass the root
directory .../mnist_runs/ as the logdir, then TensorBoard will load run1 and
run2 and compare the two for you. In general, TensorBoard will recursively
search the logdir provided, looking for subdirectories that contain TensorFlow
event data.

### Exporting Data from TensorBoard

If you'd like to export data to visualize elsewhere (e.g. iPython Notebook),
that's possible too. You can directly depend on the underlying classes that
TensorBoard uses for loading data: `python/summary/event_accumulator.py` (for
loading data from a single run) or `python/summary/event_multiplexer.py` (for
loading data from multiple runs, and keeping it organized). These classes load
groups of event files, discard data that was "orphaned" by TensorFlow crashes,
and organize the data by tag.

As another option, there is a script
(tensorboard/scripts/serialize_tensorboard.py) which will load a logdir just
like TensorBoard does, but write all of the data out to disk as json instead of
starting a server. This script is setup to make "fake TensorBoard backends" for
testing, so it is a bit rough around the edges.

### Purging 'Orphaned' Data

A TensorFlow job may occasionally crash and restart, for example if it was
preempted by a higher priority job. When this happens, some data that was
written to the summary files becomes 'orphaned' - for example, if TensorFlow ran
to step 1337, but then crashed and restarted at step 1000, the data written from
steps 1001 to 1337 is 'orphaned' - it no longer is part of the history of your
TensorFlow job.

TensorBoard attempts to detect this, and purge the orphaned data. It does this
by looking for a TensorFlow SessionLog.START event, and throwing away all data
that occurred after the new SessionLog.START. If your TensorBoard seems to be
missing valid data, it is possible that this logic is the culprit. Try launching
TensorBoard with --purge_orphaned_data=False and see if your problem persists.

# Architecture

TensorBoard consists of a Python backend (tensorboard/backend/) and a
Typescript/Polymer/D3 frontend (tensorboard/lib/, tensorboard/components).

# TensorBoard Development Instructions

The following instructions are useful if you want to develop the TensorBoard
frontend in a lightweight frontend-only environment. It sets up gulp with
automatic recompiling and serves just the frontend assets without a connected
backend.

If you just want to use TensorBoard, there is no need to read any further.

### Install Node, npm, gulp, and bower in your machine
Get nodejs and npm through whatever package distribution system is appropriate
for your machine. For example, on Ubuntu 14.04, run
`sudo apt-get install nodejs nodejs-legacy npm`. Then, run
`sudo npm install -g gulp bower`.

### Install project dependencies

Inside this directory (`tensorflow/tensorboard`),
run the following commands.

    npm install
    bower install

### Run Gulp

Inside this directory, run `gulp`. That will compile all of the
html/js/css dependencies for TensorBoard, and also spin up a server
(by default at port 8000). You can navigate to component-specific demo pages to
check out their behavior.

Running `gulp test` will run all unit tests, the linter, etc.
