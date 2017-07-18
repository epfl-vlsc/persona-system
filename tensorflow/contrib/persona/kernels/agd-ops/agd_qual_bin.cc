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
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>

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

  inline bool operator==(const Position& lhs, const Position& rhs) {
    return (lhs.ref_index() == rhs.ref_index() && lhs.position() == rhs.position());
  }

  class AGDQualBinOp : public OpKernel {
  public:
    AGDQualBinOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    ~AGDQualBinOp() {
      core::ScopedUnref unref_listpool(bufferpair_pool_);
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("marked_results", ctx));
      return Status::OK();
    }
    
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

      return Status::OK();
    }
   
    void Compute(OpKernelContext* ctx) override {
      
      if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }
      //read input Tensors
      const Tensor* results_t, *num_results_t, *upper_bounds_t, *bin_values_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      OP_REQUIRES_OK(ctx, ctx->input("upper_bounds", &upper_bounds_t));
      OP_REQUIRES_OK(ctx, ctx->input("bin_values", &bin_values_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto upper_bounds = upper_bounds_t->vec<int32>();
      auto bin_values = bin_values_t->vec<int32>();
      auto rmgr = ctx->resource_manager();
	
      //setup output container and AlignmentResultBuilder
      ResourceContainer<BufferPair> *output_bufferpair_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx, &output_bufferpair_container));
      auto output_bufferpair = output_bufferpair_container->get();
      AlignmentResultBuilder results_builder;
      results_builder.SetBufferPair(output_bufferpair);
     
      //setup Record Reader
      ResourceContainer<Data> *record_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &record_container));
      AGDRecordReader record_reader(record_container, num_results); 
      Alignment agd_record; 
      const char* record;
      size_t chunksize;
      
      Status s = record_reader.GetNextRecord(&record, &chunksize);

      while (s.ok()) {
	agd_record.ParseFromArray(record, chunksize);
	int mapping_quality = agd_record.mapping_quality();
	int num_bins = upper_bounds.size();
	
	int i = 0;	
	//find corresponding bin and change quality value
	while (i<num_bins){
		if (mapping_quality<=upper_bounds[i]){
			agd_record.set_mapping_quality(bin_values[i]);
			break;
		}
		i++;
	}
	//handle edge case
	if (i==num_bins){
		agd_record.set_mapping_quality(bin_values[num_bins-1]);
	}
	
	results_builder.AppendAlignmentResult(agd_record);
	s = record_reader.GetNextRecord(&record, &chunksize);
	
      } // while s is ok()

      //clean up
      resource_releaser(record_container);
      
    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDQualBin").Device(DEVICE_CPU), AGDQualBinOp);
} //  namespace tensorflow {
