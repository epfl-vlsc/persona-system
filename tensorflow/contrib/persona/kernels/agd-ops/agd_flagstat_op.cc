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

  class AGDFlagstatOp : public OpKernel {
  public:

    int64 count_total_reads[2]; 
    int64 count_secondary[2]; 
    int64 count_supplementary[2]; 
    int64 count_duplicates[2]; 
    int64 count_mapped[2]; 
    int64 count_paired[2]; 
    int64 count_first[2]; 
    int64 count_last[2]; 
    int64 count_properly_paired[2]; 
    int64 count_with_itself_and_mate_mapped[2];
    int64 count_singletons[2]; 
    int64 count_mate_mapped_to_diff_chr[2]; 
    int64 count_mate_mapped_to_diff_chr_mapq[2]; 


    AGDFlagstatOp(OpKernelConstruction *context) : OpKernel(context) {
      // cout<<"Starting flagstat constructor \n";
      for(int64 i = 0 ; i < 2 ; i++ )
      {
        count_total_reads[i] = 0;
        count_secondary[i] = 0; 
        count_supplementary[i] = 0; 
        count_duplicates[i] = 0; 
        count_mapped[i] = 0; 
        count_paired[i] = 0; 
        count_first[i] = 0; 
        count_last[i] = 0; 
        count_properly_paired[i] = 0; 
        count_with_itself_and_mate_mapped[i] = 0;
        count_singletons[i] = 0; 
        count_mate_mapped_to_diff_chr[i] = 0; 
        count_mate_mapped_to_diff_chr_mapq[i] = 0;
      }
    }

    ~AGDFlagstatOp() {
      // cout<<"Done flagstat destructor \n";
      cout<<count_total_reads[0]<<" + "<<count_total_reads[1]<<" in total (QC-passed reads + QC-failed reads)\n";
      cout<<count_secondary[0]<<" + "<<count_secondary[1]<<" secondaries\n";
      cout<<count_supplementary[0]<<" + "<<count_supplementary[1]<<" supplementaries\n";
      cout<<count_duplicates[0]<<" + "<<count_duplicates[1]<<" duplicates\n";
      cout<<count_mapped[0]<<" + "<<count_mapped[1]<<" mapped ( "<<(count_mapped[0]*100.0)/count_total_reads[0]<<"% : "<<(count_mapped[1]*100.0)/count_total_reads[1] <<"% )\n";
      cout<<count_paired[0]<<" + "<<count_paired[1]<<" paired in sequencing\n";
      cout<<count_first[0]<<" + "<<count_first[1]<<" first\n";
      cout<<count_last[0]<<" + "<<count_last[1]<<" last\n";
      cout<<count_properly_paired[0]<<" + "<<count_properly_paired[1]<<" properly paired ( "<<(count_properly_paired[0]*100.0)/count_total_reads[0]<<"% : "<<(count_properly_paired[1]*100.0)/count_total_reads[1] <<"% )\n";
      cout<<count_with_itself_and_mate_mapped[0]<<" + "<<count_with_itself_and_mate_mapped[1]<<" with itself and mate mapped\n";
      cout<<count_singletons[0]<<" + "<<count_singletons[1]<<" singletons ( "<<(count_singletons[0]*100.0)/count_total_reads[0]<<"% : "<<(count_singletons[1]*100.0)/count_total_reads[1] <<"% )\n";
      cout<<count_mate_mapped_to_diff_chr[0]<<" + "<<count_mate_mapped_to_diff_chr[1]<<" with mate mapped to different chr\n";
      cout<<count_mate_mapped_to_diff_chr_mapq[0]<<" + "<<count_mate_mapped_to_diff_chr_mapq[1]<<" with mate mapped to different chr (mapQ>=5)\n";
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor* results_t, *num_results_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *results_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      AGDResultReader results_reader(results_container, num_results);

      Alignment result;
      Alignment mate;
      Status s = results_reader.GetNextResult(result);
      
      while (s.ok()) {

        uint32 idx = 1;
        if ((result.flag() & ResultFlag::FILTERED) == 0)
          idx = 0;

        count_total_reads[idx]++;
        if (IsSecondary(result.flag()))
          count_secondary[idx]++;
        else if (IsSupplemental(result.flag()))
          count_supplementary[idx]++;
        else if (IsPaired(result.flag()))
        {
          count_paired[idx]++;
          if (IsFirstRead(result.flag()))
            count_first[idx]++;
          if (IsLastRead(result.flag()))
            count_last[idx]++;
          if (!IsDiscordant(result.flag()) && IsMapped(result.flag()))
            count_properly_paired[idx]++;
          if (IsNextUnmapped(result.flag()) && IsMapped(result.flag()))
            count_singletons[idx]++;
          if (IsNextMapped(result.flag()) && IsMapped(result.flag()))
          {
            count_with_itself_and_mate_mapped[idx]++;
            if(result.next_position().ref_index() != result.position().ref_index())  // MRNM != RNAME
            {
              count_mate_mapped_to_diff_chr[idx]++;
              if(result.mapping_quality() >= 5)
                count_mate_mapped_to_diff_chr_mapq[idx]++;
            }

          }
        if (IsDuplicate(result.flag()))
          count_duplicates[idx]++;
        if (IsMapped(result.flag()))
          count_mapped[idx]++;
      
        }

        s = results_reader.GetNextResult(result);
      } // while s is ok()
      // Grab the input tensor
      const Tensor& input_tensor = ctx->input(1);
      auto input = input_tensor.flat<int32>();
      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),&output_tensor));
      auto output = output_tensor->flat<int32>();
      // cout<<"Filling values";
      const int N = input.size();
      for (int i = 0; i < N; i++) {
        output(i) = 0;
      }
      // done
      resource_releaser(results_container);
    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDFlagstat").Device(DEVICE_CPU), AGDFlagstatOp);
} //  namespace tensorflow {
