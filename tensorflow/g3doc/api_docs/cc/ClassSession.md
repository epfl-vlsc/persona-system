# `class tensorflow::Session`

A Session instance lets a caller drive a TensorFlow graph computation.

When a Session is created with a given target, a new Session object is bound to the universe of resources specified by that target. Those resources are available to this session to perform computation described in the GraphDef. After extending the session with a graph, the caller uses the Run() API to perform the computation and potentially fetch outputs as Tensors.

Example:

```c++ tensorflow::GraphDef graph;
// ... Create or load graph into "graph".

// This example uses the default options which connects
// to a local runtime.
tensorflow::SessionOptions options;
std::unique_ptr<tensorflow::Session>
session(tensorflow::NewSession(options));

// Create the session with this graph.
tensorflow::Status s = session->Create(graph);
if (!s.ok()) { ... }

// Run the graph and fetch the first output of the "output"
// operation, and also run to but do not return anything
// for the "update_state" operation.
std::vector<tensorflow::Tensor> outputs;
s = session->Run({}, {"output:0"}, {"update_state"}, &outputs);
if (!s.ok()) { ... }

// Map the output as a flattened float tensor, and do something
// with it.
auto output_tensor = outputs[0].flat<float>();
if (output_tensor(0) > 0.5) { ... }

// Close the session to release the resources associated with
// this session.
session->Close();

```

A Session allows concurrent calls to Run() , though a Session must be created / extended by a single thread.

Only one thread must call Close() , and Close() must only be called after all other calls to Run() have returned.

###Member Details

#### `virtual Status tensorflow::Session::Create(const GraphDef &graph)=0` {#virtual_Status_tensorflow_Session_Create}

Create the graph to be used for the session.

Returns an error if this session has already been created with a graph. To re-use the session with a different graph, the caller must Close() the session first.

#### `virtual Status tensorflow::Session::Extend(const GraphDef &graph)=0` {#virtual_Status_tensorflow_Session_Extend}

Adds operations to the graph that is already registered with the Session .

The names of new operations in "graph" must not exist in the graph that is already registered.

#### `virtual Status tensorflow::Session::Run(const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names, const std::vector< string > &target_node_names, std::vector< Tensor > *outputs)=0` {#virtual_Status_tensorflow_Session_Run}

Runs the graph with the provided input tensors and fills `outputs` for the endpoints specified in `output_tensor_names`. Runs to but does not return Tensors for the nodes in `target_node_names`.

The order of tensors in `outputs` will match the order provided by `output_tensor_names`.

If `Run` returns `OK()`, then `outputs->size()` will be equal to `output_tensor_names.size()`. If `Run` does not return `OK()`, the state of `outputs` is undefined.

REQUIRES: The name of each Tensor of the input or output must match a "Tensor endpoint" in the `GraphDef` passed to ` Create() `.

REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.

#### `virtual Status tensorflow::Session::RunWithOpts(const RunOptions &run_options, const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names, const std::vector< string > &target_node_names, std::vector< Tensor > *outputs, RunOutputs *run_outputs)` {#virtual_Status_tensorflow_Session_RunWithOpts}

Like `Run`, but allows users to pass in a `RunOptions` proto and to retrieve non-Tensor metadata output via a `RunOutputs` proto for this step. NOTE: This API is still experimental and may change.



#### `virtual Status tensorflow::Session::PRunSetup(const std::vector< string > &input_names, const std::vector< string > &output_names, const std::vector< string > &target_nodes, string *handle)` {#virtual_Status_tensorflow_Session_PRunSetup}

Sets up a graph for partial execution. All future feeds and fetches are specified by &apos;input_names&apos; and &apos;output_names&apos;. Returns &apos;handle&apos; that can be used to perform a sequence of partial feeds and fetches. NOTE: This API is still experimental and may change.



#### `virtual Status tensorflow::Session::PRun(const string &handle, const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_names, std::vector< Tensor > *outputs)` {#virtual_Status_tensorflow_Session_PRun}

Continues the pending execution specified by &apos;handle&apos; with the provided input tensors and fills `outputs` for the endpoints specified in `output_names`. NOTE: This API is still experimental and may change.



#### `virtual Status tensorflow::Session::Close()=0` {#virtual_Status_tensorflow_Session_Close}

Closes this session.

Closing a session releases the resources used by this session on the TensorFlow runtime (specified during session creation by the ` SessionOptions::target ` field).

#### `virtual tensorflow::Session::~Session()` {#virtual_tensorflow_Session_Session}




