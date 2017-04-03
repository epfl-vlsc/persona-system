//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "compression.h"
#include "parser.h"
#include "util.h"
#include "buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

namespace tensorflow {


  using namespace std;
  using namespace errors;

  class AGDReaderOp : public OpKernel {
  public:
    AGDReaderOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("verify", &verify_));
      OP_REQUIRES_OK(context, context->GetAttr("twobit", &twobit_));
      if (verify_) {
        LOG(INFO) << name() << " enabled verification\n";
      }

      int32_t i;
      OP_REQUIRES_OK(context, context->GetAttr("reserve", &i));
      reserve_bytes_ = static_cast<decltype(reserve_bytes_)>(i);
      
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
    }

    ~AGDReaderOp() {
      core::ScopedUnref unref_pool(buffer_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
      }

      const Tensor *file_handle_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &file_handle_t));
      auto fileset = file_handle_t->vec<string>();

      Tensor *num_records_t, *first_ordinals_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", scalar_shape_, &num_records_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", scalar_shape_, &first_ordinals_t));
      auto num_records = num_records_t->scalar<int32>();
      auto first_ordinals = first_ordinals_t->scalar<int64>();

      // ALl output is set up at this point

      ResourceContainer<Data> *agd_input;
      ResourceContainer<Buffer> *output_buffer_rc;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &agd_input));
      core::ScopedUnref unref_me(agd_input);
      {
        ResourceReleaser<Data> agd_releaser(*agd_input);
        auto input_data = agd_input->get();
        {
          DataReleaser dr(*input_data);
          OP_REQUIRES_OK(ctx, buffer_pool_->GetResource(&output_buffer_rc));
          auto* output_ptr = output_buffer_rc->get();
          output_ptr->reserve(250*1024*1024); // make sure its big enough if its fresh.

          OP_REQUIRES_OK(ctx, rec_parser_.ParseNew(input_data->data(), input_data->size(),
                                                   verify_, output_ptr, &first_ord_, &num_recs_, unpack_, twobit_));
          OP_REQUIRES_OK(ctx, output_buffer_rc->allocate_output("processed_buffers", ctx));
          num_records() = num_recs_;
          first_ordinals() = first_ord_;
        }
      }
    }

  private:
    bool verify_ = false;
    bool twobit_ = false;
    RecordParser rec_parser_;
    size_t reserve_bytes_;
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
    bool unpack_ = true;
    const TensorShape scalar_shape_{};

    uint32_t num_recs_;
    uint64_t first_ord_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDReader").Device(DEVICE_CPU), AGDReaderOp);
} //  namespace tensorflow {
