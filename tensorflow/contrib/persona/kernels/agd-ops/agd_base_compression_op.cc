#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <string>
#include <iostream>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
//alignment.pb.h is a special cmd.
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"

/*
  This is the code for a base compression operator.
  ABH 2018
*/

namespace tensorflow {

  using namespace std;
  using namespace errors;

  namespace {
     void resource_releaser(ResourceContainer<Data> *data) {
       core::ScopedUnref a(data);
       data->release();
     }
  }

  class AGDBaseCompressionOp : public OpKernel {
  public:
    //Constructor
    AGDBaseCompressionOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
    }
    //our main function of compression.
    void Compute(OpKernelContext* ctx) override {
      LOG(INFO) << "Starting compression";
      //create 2 tensor : 1 ressource and 1 chunk size
      //still
      const Tensor *results_t, *chunk_size_t;
      Tensor *ret_t;
      //this should point to the first element of or resources that should be 0.
      //"tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
      ResourceContainer<Data> *results_container;

      // For this given line arg comes from agd_merge_op line 230.
      // OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &results_container));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      OP_REQUIRES_OK(ctx, ctx->input("results", &results_t));
      //TODO in case of error create an output tensor and put some value in it.
      auto results = results_t->vec<string>();
      auto chunk_size = chunk_size_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();

      /*just to have any output */
      OP_REQUIRES_OK(ctx, ctx->allocate_output("ret",scalar_shape_,&ret_t));
      ret_t->scalar<int32>()() = 0;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results(0), results(1), &results_container));

      Alignment agd_result;
      //results reader allow to get acces to the data inside the container.
      AGDResultReader results_reader(results_container, chunk_size);
      Status s = results_reader.GetNextResult(agd_result);

      while(s.ok()){
        LOG(INFO) << "results flag : " << agd_result.flag();
        //TODO format note good for print.
        LOG(INFO) << "results position : " << agd_result.position().position();
        s = results_reader.GetNextResult(agd_result);
      }

      resource_releaser(results_container);
    }


  private:
    // RecordParser rec_parser_;
    const TensorShape scalar_shape_{};
    string record_id_;
    bool unpack_;
    vector<string> columns_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDBaseCompression").Device(DEVICE_CPU), AGDBaseCompressionOp);
} //  namespace tensorflow {
