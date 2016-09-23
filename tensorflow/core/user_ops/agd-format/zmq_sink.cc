#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"

// Needed for mkfifo and other low-level stuff
#include <zmq.hpp>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("ZeroMqSink");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("url: string")
  .Input("input: string")
  .Doc(R"doc(
Creates a zmq writer that sends it's input to the specified URL
)doc");

  class ZeroMqSinkOp : public OpKernel {
  public:
    ZeroMqSinkOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("url", &url_));
      init();
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("input", &input));
      auto msg_str = input->scalar<string>()();

      LOG(INFO) << "Sending string: " << msg_str;

      auto msg_sz = msg_str.size();
      zmq::message_t msg(msg_sz);
      memcpy(msg.data(), msg_str.c_str(), msg_sz);

      // TODO in the future, we should be smart and retry connecting
      OP_REQUIRES(ctx, socket_->send(msg), Internal("sending message to '", url_, "' failed"));
    }

  private:

    Status init() {

      context_.reset(new zmq::context_t(1));
      socket_.reset(new zmq::socket_t(*context_, ZMQ_PUSH));
      socket_->connect(url_.c_str());

      return Status::OK();
    }

    unique_ptr<zmq::context_t> context_;
    unique_ptr<zmq::socket_t> socket_;
    string url_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ZeroMqSinkOp);
} // namespace tensorflow {
