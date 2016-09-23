#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"

// Needed for mkfifo and other low-level stuff
#include <zmq.hpp>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("ZeroMqSource");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("url: string")
  .Output("output: string")
  .Doc(R"doc(
  Creates a ZMQ reader that reads one line at a time from a ZMQ url of form tcp://blah:1234
)doc");

  class ZeroMqSourceOp : public OpKernel {
  public:
    ZeroMqSourceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("url", &url_));
      init();
    }

    ~ZeroMqSourceOp() override {
      delete context_;
    }

    void Compute(OpKernelContext* ctx) override {

      zmq::message_t msg;
      socket_->recv(&msg);
      OP_REQUIRES(ctx, msg.size() > 0, Internal("ZeroMqSourceOp received message of size 0!")); 
      LOG(INFO) << "zmq source received input: " << (char *)msg.data();
      string input((char *)msg.data());

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("output", TensorShape({}), &output));
      output->scalar<string>()() = input;
    }

  private:

    Status init() {

      context_ = new zmq::context_t(1);
      socket_.reset(new zmq::socket_t(*context_, ZMQ_REP));
      socket_->bind(url_.c_str());

      return Status::OK();
    }

    zmq::context_t* context_;
    unique_ptr<zmq::socket_t> socket_;
    string url_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ZeroMqSourceOp);
} // namespace tensorflow {
