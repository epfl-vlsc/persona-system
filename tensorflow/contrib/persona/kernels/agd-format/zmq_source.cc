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

  class ZeroMqSourceOp : public OpKernel {
  public:
    ZeroMqSourceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("url", &url_));
      init();
    }

    void Compute(OpKernelContext* ctx) override {
      zmq::message_t msg;
      socket_->recv(&msg);
      OP_REQUIRES(ctx, msg.size() > 0, Internal("ZeroMqSourceOp received message of size 0!"));
      string input((char *)msg.data(), msg.size());

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("output", TensorShape({}), &output));
      output->scalar<string>()() = input;
    }

  private:

    Status init() {

      // no default constructor
      context_.reset(new zmq::context_t(1));
      socket_.reset(new zmq::socket_t(*context_, ZMQ_PULL));
      socket_->connect(url_.c_str());

      return Status::OK();
    }

    unique_ptr<zmq::context_t> context_;
    unique_ptr<zmq::socket_t> socket_;
    string url_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ZeroMqSourceOp);
} // namespace tensorflow {
