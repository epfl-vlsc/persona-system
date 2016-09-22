#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"

// Needed for mkfifo and other low-level stuff
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("PipeSource");
    const size_t line_buf_size = 1024; // a sensible initial size
  }

  REGISTER_OP(op_name.c_str())
  .Attr("path: string")
  .Attr("create: bool = false")
  .Output("output: string")
  .Doc(R"doc(
Opens a pipe at `path`, and emits newline-delimited strings (one line at a time) until it receives feof, then throwing a stop exception.
)doc");

  class PipeSourceOp : public OpKernel {
  public:
    PipeSourceOp(OpKernelConstruction *ctx) : OpKernel(ctx), buf_sz_(line_buf_size), line_buf_(new char[line_buf_size]) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &path_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("create", &create_));
    }

    ~PipeSourceOp() override {
      int ret;
      if (fifo_fp_ != nullptr) {
        ret = fclose(fifo_fp_);
        if (ret != 0) {
          LOG(ERROR) << "Calling fclose on path " << path_ << " returned value " << ret;
        }
      }

      if (create_) {
        // TODO delete the fifo
      }

      delete [] line_buf_;
    }

    void Compute(OpKernelContext* ctx) override {
      if (!fifo_fp_) {
        OP_REQUIRES_OK(ctx, init());
      }
      auto ret = getline(&line_buf_, &buf_sz_, fifo_fp_);
      if (ret < 0) {
        Status s;
        if (feof(fifo_fp_)) {
          s = OutOfRange("end of file stream for named pipe at ", path_);
        } else {
          s = Internal("errno value of ", ret, " return for file stream at ", path_);
        }
        ctx->SetStatus(s);
        return;
      }

      OP_REQUIRES(ctx, ret != -1, Internal("getline() on named pipe '", path_, "' returned ", ret));
      // TODO check errno to shut down correctly
      string s(line_buf_, ret-1); // -1 to not include the delim character

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("output", TensorShape({}), &output));
      output->scalar<string>()() = s;
    }

  private:

    Status init() {
      int status;

      if (create_) {
        status = mkfifo(path_.c_str(), S_IRUSR | S_IWUSR);
        if (status != 0) {
          return Internal("mkfifo() on path '", path_, "' returns error ", status);
        }
      }

      fifo_fp_ = fopen(path_.c_str(), "r");
      if (!fifo_fp_) {
        return Internal("fdopen for fifo at path '", path_, "' failed");
      }

      return Status::OK();
    }

    string path_;
    char *line_buf_;
    size_t buf_sz_;
    FILE *fifo_fp_ = nullptr;
    bool create_ = false;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), PipeSourceOp);
} // namespace tensorflow {
