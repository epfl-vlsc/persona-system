#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "parser.h"

namespace tensorflow {
  using namespace std;

  REGISTER_REFERENCE_POOL("ParserPool")
  .Attr("size_hint: int = 4194304") // 4 MeB
  .Doc(R"doc(
Creates and initializes a pool containing the `size` number of RecordParser objects
)doc");

  class ParserPoolOp : public ReferencePoolOp<RecordParser, RecordParser> {
  public:
    ParserPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<RecordParser, RecordParser>(ctx) {
      using namespace errors;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("size_hint", &size_hint_));
      OP_REQUIRES(ctx, size_hint_ > 0, InvalidArgument("ParserPoolOp requires size_hint > 0 : ", size_hint_));
    }

  protected:
    unique_ptr<RecordParser> CreateObject() override {
      return unique_ptr<RecordParser>(new RecordParser(size_hint_));
    }

  private:
    int size_hint_;
  };

  REGISTER_KERNEL_BUILDER(Name("ParserPool").Device(DEVICE_CPU), ParserPoolOp);
} // namespace tensorflow {
