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
#include <string>
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

  class ImportSGAOp : public OpKernel {
  public:
    ImportSGAOp(OpKernelConstruction *context) : OpKernel(context) {
	string path;
	OP_REQUIRES_OK(context, context->GetAttr("path", &path));
        OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(context, context->GetAttr("ref_seq_sizes", &ref_sizes_));
        OP_REQUIRES_OK(context, context->GetAttr("feature", &feature_));
        OP_REQUIRES(context, ref_seqs_.size() == ref_sizes_.size(), Internal("ref seqs was not same size as ref seq sizes lists"));

        // flag is used to generate output array only for those references which have some alignment in the read chunks
        for(int i=0;i<ref_seqs_.size();i++)
        {
          flag.push_back(-1);
        }
	
        //a 2-d array to store number of alignments
        startRead = -1;
        lenRead = -1;
        strandRead = "";
	
	if(path.back() == '/'){
           path=path+"output.sga";
        }else{
	   path = path+"/output.sga";
        }
        fp = fopen( path.c_str(), "w" );
 

    }

    ~ImportSGAOp()
    {
      if(countp != 0)
        fprintf(fp, "%s\t%s\t%u\t + \t%u\n",ref_seqs_[prevIndex].c_str(), feature_.c_str(), startRead+1, countp);
      if(countn != 0)
        fprintf(fp, "%s\t%s\t%u\t - \t%u\n",ref_seqs_[prevIndex].c_str(), feature_.c_str(), startRead+lenRead, countn);
      fclose(fp);
    }


    //for the cigar string find the op name and number
    inline int parseNextOp(const char *ptr, char &op, int &num)
    {
      num = 0;
      const char * begin = ptr;
      for (char curChar = ptr[0]; curChar != 0; curChar = (++ptr)[0])
      {
        int digit = curChar - '0';
        if (digit >= 0 && digit <= 9) num = num*10 + digit;
        else break;
      }
      op = (ptr++)[0];
      return ptr - begin;
    }


    // main function to increment the counter for each result in the agd column
    Status CalculateCoverage(const Alignment *result,uint32 flagf) {

      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      int readLength = 0;
      char op;
      int op_len;
      int index = result->position().ref_index();

      if(flag[index]==-1)
      {
        flag[index]=0;
      }
      int start = result->position().position();
      char *p = (char*) cigar;
      while (*p) { // While there are more characters to process...
        if (isdigit(*p)) { // Upon finding a digit, ...
          long val = strtol(p, &p, 10); // Read a number, ...
          readLength += val;
        } else { // Otherwise, move on to the next character.
          p++;
        }
      }      

      string strand;
      if(IsForwardStrand(flagf))
        strand = "+";
      else
        strand = "-";


      if(startRead == -1){
        startRead = start;
        countp = 0;
        countn = 0;
        lenRead = readLength;
        strandRead = strand;
      }

      //works only if sorted
      if(startRead == start && lenRead == readLength) {
        if(strand == "+")
          countp++;
        else
          countn++;
        prevIndex = index;
      } else {
          if(countp != 0)
             fprintf(fp, "%s\t%s\t%u\t + \t%u\n",ref_seqs_[prevIndex].c_str(),feature_.c_str(), startRead+1, countp);
          if(countn != 0)
             fprintf(fp, "%s\t%s\t%u\t - \t%u\n",ref_seqs_[prevIndex].c_str(), feature_.c_str(), startRead+lenRead, countn);
        startRead = start;
        lenRead = readLength;
        countp = 0;
        countn = 0;
        if(strand == "+")
          countp++;
        else
          countn++;
        prevIndex = index;
        strandRead = strand;
      }  
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {

      //compute = 1;
      const Tensor* results_t, *num_results_t,*scale_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *results_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      AGDResultReader results_reader(results_container, num_results);
      const Tensor& input_tensor = ctx->input(1);
      auto input = input_tensor.flat<int32>();


      // Create an output tensor
      // no need for output tensor , its a dummy tensor
      // TODO can remove this output tensor but something must be fed into queue it gives output to
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto outputa = output_tensor->flat<int32>();

      // Set all but the first element of the output tensor to 0.
      const int N = input.size();
      for (int i = 0; i < N; i++) {
        outputa(i) = 0;
      }





      Alignment result;
      Status s = results_reader.GetNextResult(result);
      
      while (s.ok()) {
        if (!IsPrimary(result.flag()))
          OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
                result.position().DebugString()));

          // we have a single alignment
          if (IsUnmapped(result.flag())) {

            fflush(stdout);
            s = results_reader.GetNextResult(result);
            continue;
          }
          OP_REQUIRES_OK(ctx, CalculateCoverage(&result,result.flag()));
        s = results_reader.GetNextResult(result);

        fflush(stdout);
      } // while s is ok()
      resource_releaser(results_container);
    }

  private:
    vector<string> ref_seqs_;
    vector<int32> ref_sizes_;
    vector<int> flag;
    int startRead, lenRead, countp, countn, prevIndex;
    string strandRead;
    FILE * fp;
    string feature_;

  };

  REGISTER_KERNEL_BUILDER(Name("ImportSGA").Device(DEVICE_CPU), ImportSGAOp);
} //  namespace tensorflow {
