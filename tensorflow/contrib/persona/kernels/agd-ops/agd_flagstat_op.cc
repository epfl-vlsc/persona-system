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
      cout<<"Starting flagstat constructor \n";
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
      cout<<"Done flagstat destructor \n";
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

      // core::ScopedUnref unref_listpool(bufferpair_pool_);
    }

    // Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    // {
    //   TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
    //   (*ctr)->get()->reset();
    //   TF_RETURN_IF_ERROR((*ctr)->allocate_output("marked_results", ctx));
    //   return Status::OK();
    // }
    
    // Status InitHandles(OpKernelContext* ctx)
    // {
    //   TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

    //   return Status::OK();
    // }

    // inline int parseNextOp(const char *ptr, char &op, int &num)
    // {
    //   num = 0;
    //   const char * begin = ptr;
    //   for (char curChar = ptr[0]; curChar != 0; curChar = (++ptr)[0])
    //   {
    //     int digit = curChar - '0';
    //     if (digit >= 0 && digit <= 9) num = num*10 + digit;
    //     else break;
    //   }
    //   op = (ptr++)[0];
    //   return ptr - begin;
    // }
   
    // Status CalculatePosition(const Alignment *result,
    //     uint32_t &position) {
    //   // figure out the 5' position
    //   // the result->location is already genome relative, we shouldn't have to worry 
    //   // about negative index after clipping, but double check anyway
    //   // cigar parsing adapted from samblaster
    //   const char* cigar;
    //   size_t cigar_len;
    //   cigar = result->cigar().c_str();
    //   cigar_len = result->cigar().length();
    //   int ralen = 0, qalen = 0, sclip = 0, eclip = 0;
    //   bool first = true;
    //   char op;
    //   int op_len;
    //   while(cigar_len > 0) {
    //     size_t len = parseNextOp(cigar, op, op_len);
    //     cigar += len;
    //     cigar_len -= len;         
    //     //LOG(INFO) << "cigar was " << op_len << " " << op;
    //     //LOG(INFO) << "cigar len is now: " << cigar_len;
    //     if (op == 'M' || op == '=' || op == 'X')
    //     {
    //       ralen += op_len;
    //       qalen += op_len;
    //       first = false;
    //     }
    //     else if (op == 'S' || op == 'H')
    //     {
    //       if (first) sclip += op_len;
    //       else       eclip += op_len;
    //     }
    //     else if (op == 'D' || op == 'N')
    //     {
    //       ralen += op_len;
    //     }
    //     else if (op == 'I')
    //     {
    //       qalen += op_len;
    //     }
    //     else
    //     {
    //       return Internal("Unknown opcode ", string(&op, 1), " in CIGAR string: ", string(cigar, cigar_len));
    //     }
    //   }
    //   //LOG(INFO) << "the location is: " << result->location_;
    //   if (IsForwardStrand(result->flag())) {
    //     position = static_cast<uint32_t>(result->location() - sclip);
    //   } else {
    //     // im not 100% sure this is correct ...
    //     // but if it goes for every signature then it shouldn't matter
    //     position = static_cast<uint32_t>(result->location() + ralen + eclip - 1);
    //   }
    //   //LOG(INFO) << "position is now: " << position;
    //   if (position < 0)
    //     return Internal("A position after applying clipping was < 0! --> ", position);
    //   return Status::OK();
    // }

    // Status MarkDuplicate(const Alignment* result, AlignmentResultBuilder &builder) {
    //   Alignment result_out;
    //   result_out.CopyFrom(*result); // simple copy suffices
    //   result_out.set_flag(result_out.flag()| ResultFlag::PCR_DUPLICATE);
    //   // yes we are copying and rebuilding the entire structure
    //   // modifying in place is a huge pain in the ass, and the results aren't that
    //   // big anyway
    //   builder.AppendAlignmentResult(result_out);
    //   return Status::OK();
    // }

    // Status ProcessOrphan(const Alignment* result, AlignmentResultBuilder &builder) {
    //   Signature sig;
    //   sig.is_forward = IsForwardStrand(result->flag());
    //   TF_RETURN_IF_ERROR(CalculatePosition(result, sig.position));

    //   //LOG(INFO) << "sig is: " << sig.ToString();
    //   // attempt to find the signature
    //   auto sig_map_iter = signature_map_->find(sig);
    //   if (sig_map_iter == signature_map_->end()) { // not found, insert it
    //     signature_map_->insert(make_pair(sig, 1));
    //     // its the first here, others will be marked dup
    //     builder.AppendAlignmentResult(*result);
    //     return Status::OK();
    //   } else { 
    //     // found, mark a dup
    //     num_dups_found_++;
    //     return MarkDuplicate(result, builder);
    //   }
    // }

    void Compute(OpKernelContext* ctx) override {
      // if (!bufferpair_pool_) {
      //   OP_REQUIRES_OK(ctx, InitHandles(ctx));
      // }

      // LOG(INFO) << "Starting flagstat";
      const Tensor* results_t, *num_results_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *results_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      AGDResultReader results_reader(results_container, num_results);

      // get output buffer pairs (pair holds [index, data] to construct
      // the results builder for output
      // ResourceContainer<BufferPair> *output_bufferpair_container;
      // OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx, &output_bufferpair_container));
      // auto output_bufferpair = output_bufferpair_container->get();
      // AlignmentResultBuilder results_builder;
      // results_builder.SetBufferPair(output_bufferpair);


      Alignment result;
      Alignment mate;
      Status s = results_reader.GetNextResult(result);

      
      

      while (s.ok()) {

        // cout<<"Flag : "<<result.flag()<<"\n";

        uint32 idx = 1;
        if ((result.flag() & ResultFlag::FILTERED) == 0)
          idx = 0;

        count_total_reads[idx]++;
        // cout<<";";
        if (IsSecondary(result.flag()))
          count_secondary[idx]++;
        else if (IsSupplemental(result.flag()))
          count_supplementary[idx]++;
        else if (IsPaired(result.flag()))
        {
          // cout<<"zxc";
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
            // OP_REQUIRES_OK(ctx, results_reader.GetNextResult(mate));
            // if(IsFirstRead(result.flag()))
            // {
            //   s = results_reader.GetNextResult(mate);
            //   // cout<<"_";
            //   if(mate.position().ref_index() != result.position().ref_index())  // MRNM != RNAME
            //   {
            //     count_mate_mapped_to_diff_chr[idx]+=2;
            //     if(result.mapping_quality() >= 5)
            //       count_mate_mapped_to_diff_chr_mapq[idx]+=2;
            //   }
            //   result = mate;
            //   continue;
            // }
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

        // cout<<"qwe";
        s = results_reader.GetNextResult(result);
      } // while s is ok()

      // cout<<"Out of while loop, Printing stats\n";
      


        // if (!IsPrimary(result.flag()))
        //   OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
        //         result.location()));
        // if (!IsPaired(result.flag())) {
        //   // we have a single alignment
        //   if (IsUnmapped(result.flag())) {
        //     s = results_reader.GetNextResult(result);
        //     continue;
        //   }

        //   //LOG(INFO) << "processing mapped orphan at " << result->location_;
        //   OP_REQUIRES_OK(ctx, ProcessOrphan(&result, results_builder));

        // } else { // we have a pair, get the mate
        //   OP_REQUIRES_OK(ctx, results_reader.GetNextResult(mate));

        //   OP_REQUIRES(ctx, (result.next_location() == mate.location()) && (mate.next_location() == result.location()),
        //       Internal("Malformed pair or the data is not in metadata (QNAME) order. At index: ", results_reader.GetCurrentIndex()-1,
        //         "result 1: ", result.DebugString(), " result2: ", mate.DebugString()));

        //   //LOG(INFO) << "processing mapped pair at " << result->location_ << ", " << mate->location_;

        //   if (IsUnmapped(result.flag()) && IsUnmapped(mate.flag())) {
        //     s = results_reader.GetNextResult(result);
        //     continue;
        //   }
          
        //   if (IsUnmapped(result.flag()) && IsMapped(mate.flag())) { // treat as single
        //     OP_REQUIRES_OK(ctx, ProcessOrphan(&mate, results_builder));
        //   } else if (IsUnmapped(mate.flag()) && IsMapped(result.flag())) {
        //     OP_REQUIRES_OK(ctx, ProcessOrphan(&result, results_builder));
        //   } else {
        //     Signature sig;
        //     sig.is_forward = IsForwardStrand(result.flag());
        //     sig.is_mate_forward = IsForwardStrand(mate.flag());
        //     OP_REQUIRES_OK(ctx, CalculatePosition(&result, sig.position));
        //     OP_REQUIRES_OK(ctx, CalculatePosition(&mate, sig.position_mate));

        //     // attempt to find the signature
        //     auto sig_map_iter = signature_map_->find(sig);
        //     if (sig_map_iter == signature_map_->end()) { // not found, insert it
        //       signature_map_->insert(make_pair(sig, 1));
        //       results_builder.AppendAlignmentResult(result);
        //       results_builder.AppendAlignmentResult(mate);
        //     } else { 
        //       // found, mark a dup
        //       LOG(INFO) << "omg we found a duplicate";
        //       OP_REQUIRES_OK(ctx, MarkDuplicate(&result, results_builder));
        //       OP_REQUIRES_OK(ctx, MarkDuplicate(&mate, results_builder));
        //       num_dups_found_++;
        //     }
        //   }
        // }
      // cout<<"grabbing input tensor";
      // Grab the input tensor
      const Tensor& input_tensor = ctx->input(1);
      auto input = input_tensor.flat<int32>();
      // cout<<"creating output tensor";
      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),&output_tensor));
      auto output = output_tensor->flat<int32>();
      // cout<<"Filling values";
      const int N = input.size();
      for (int i = 0; i < N; i++) {
        output(i) = 0;
      }
      // cout<<"done\n";
      // done
      resource_releaser(results_container);
      // LOG(INFO) << "DONE Printing stats";

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;


  };

  REGISTER_KERNEL_BUILDER(Name("AGDFlagstat").Device(DEVICE_CPU), AGDFlagstatOp);
} //  namespace tensorflow {
