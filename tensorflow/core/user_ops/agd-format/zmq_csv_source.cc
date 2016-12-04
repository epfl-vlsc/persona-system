#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"

// Needed for mkfifo and other low-level stuff
#include <zmq.hpp>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("ZeroMqCSVSource");
    const char* const csv_delim = " ,";
  }

  REGISTER_OP(op_name.c_str())
  .Attr("url: string")
  .Attr("columns: int >= 1")
  .Output("output: string")
  .SetIsStateful()
  .Doc(R"doc(
  Creates a ZMQ reader that reads CSV line at a time from a ZMQ url of form tcp://blah:1234

  Op will pad or clip the CSV line to be exactly `columns` in terms of the length of `output`

  This dimension is specified by `columns`.
)doc");

  class ZeroMqCSVSourceOp : public OpKernel {
  public:
    ZeroMqCSVSourceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("url", &url_));
      // No need to double check for >=1. Taken care of in the opdef above
      int columns;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns));
      out_shape_.AddDim(columns);
      // Just to check whether I actually am setting this correctly
      OP_REQUIRES(ctx, out_shape_ == TensorShape({columns}), Internal("Tensorshape is not correct: ", out_shape_.DebugString()));
      init();
    }

    void Compute(OpKernelContext* ctx) override {
      zmq::message_t msg;
      socket_->recv(&msg);
      auto msg_size = msg.size();
      OP_REQUIRES(ctx, msg_size > 0, Internal("ZeroMqCSVSourceOp received message of size 0!"));
      // Note: this copy out of the buffer is needed because none of the scanning ops for c strings
      // work on non-null-delimited strings, which is how zmq sends things across the wire
      string input((char *)msg.data(), msg_size);

      // safe: string is null-terminated, and strtok requires non-const
      auto *msg_str = const_cast<char*>(input.c_str());

      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("output", out_shape_, &output_tensor));
      auto output = output_tensor->vec<string>();

      size_t i = 0;
      auto *record = strtok(msg_str, csv_delim);

      for (; i < out_shape_.dim_size(0) && record != nullptr; ++i) {
        output(i) = record; // TODO need to put this in a C++ string?
        record = strtok(nullptr, csv_delim);
      }
      for (; i < out_shape_.dim_size(0); ++i) {
        // pad with empty strings
        output(i) = "";
      }
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
    TensorShape out_shape_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ZeroMqCSVSourceOp);
} // namespace tensorflow {
