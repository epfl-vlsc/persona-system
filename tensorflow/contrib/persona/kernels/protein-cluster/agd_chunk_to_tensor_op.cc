

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }


  using namespace std;
  using namespace errors;
  using namespace format;

  // converts an AGD chunk resource into a scalar string tensor (string is basically a buffer)
  // mostly so we can rely on TF to pipe between servers
  class AGDChunkToTensorOp : public OpKernel {
  public:
    AGDChunkToTensorOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext* ctx) override {
      LOG(INFO) << "Starting chunk to tensor";

      const Tensor* chunk_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk", &chunk_t));
      auto chunk_handle = chunk_t->vec<string>();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *chunk_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(chunk_handle(0), chunk_handle(1), &chunk_container));
      //AGDRecordReader seqs_reader(seqs_container, recs);
    
      auto base_data = chunk_container->get()->data();
      Tensor* data_out_t = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("tensor_out", TensorShape(),
            &data_out_t));
      auto data_out = data_out_t->scalar<string>();
      data_out() = string(base_data, chunk_container->get()->size());

      resource_releaser(chunk_container);

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;


  };

  REGISTER_KERNEL_BUILDER(Name("AGDChunkToTensor").Device(DEVICE_CPU), AGDChunkToTensorOp);
} //  namespace tensorflow {
