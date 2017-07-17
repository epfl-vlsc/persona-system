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

  class AGDPeakDetectionOp : public OpKernel {
  public:
    AGDPeakDetectionOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(context, context->GetAttr("ref_seq_sizes", &ref_sizes_));
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
        OP_REQUIRES_OK(context, context->GetAttr("bg", &bg_));
        OP_REQUIRES_OK(context, context->GetAttr("d", &d_));
        OP_REQUIRES_OK(context, context->GetAttr("dz", &dz_));
        OP_REQUIRES_OK(context, context->GetAttr("strand", &strand_));
        OP_REQUIRES_OK(context, context->GetAttr("bga", &bga_));
        OP_REQUIRES(context, ref_seqs_.size() == ref_sizes_.size(), Internal("ref seqs was not same size as ref seq sizes lists"));

        // flag is used to generate output array only for those references which have some alignment in the read chunks
        for(int i=0;i<ref_seqs_.size();i++)
        {
          flag.push_back(-1);
        }

        //a 2-d array to store number of alignments
        output = new int*[ref_seqs_.size()];



    }
    // reporting per-base genome coverage
    void print_res_d_or_dz(bool z_)
    {
      for(int i=0;i<ref_sizes_.size();i++)
      {
        if(flag[i]==0)
        {
          for(int j=0 ; j<ref_sizes_[i];j++)
          {
            if(output[i][j]!=0 || z_)
              cout << ref_seqs_[i] << "\t" << j+1 << "\t"<< output[i][j] << endl;
          }
        }
        else if(z_)
        {
          for(int j=0 ; j<ref_sizes_[i];j++)
          {
            cout << ref_seqs_[i] << "\t" << j+1 << "\t"<< 0 << endl;
          }
        }
      }
    }

    /* output in bedgraph format(zeroes as well)
     * z_ set to 0 for bg and 1 for bga */ 
    void print_res_bg_or_bga(bool z_)
    {
      for(int i=0;i<ref_seqs_.size();i++)
      {
        if(flag[i]==0)
        {
          int lastindex = 0;
          int lastoutput = output[i][0];
          for(int j=1;j<ref_sizes_[i];j++)
          { 
            if(output[i][j]==lastoutput)
            {
              continue;
            }
            else
            {
              if(lastoutput!=0 || z_){
                cout << ref_seqs_[i]<<"\t"<< lastindex << "\t"<< j << "\t"<< lastoutput * scale_ << endl;
              }

              lastindex = j ;
              lastoutput = output[i][j];
            }
          }

          if(lastoutput!=0 || z_)
            cout << ref_seqs_[i]<<"\t"<< lastindex << "\t"<< ref_sizes_[i] << "\t"<< lastoutput * scale_ << endl;

        }
        else if(z_)
        {
          cout << ref_seqs_[i]<<"\t"<< 0 << "\t"<< ref_sizes_[i] << "\t"<< 0 * scale_ << endl;
        }
      }

    }

    // print output in default format ie through histogram
    void print_res_hist()
    {
      long long int size_it=0;
      int maxofmaxcov = -1;
      for(int k=0;k<ref_sizes_.size();k++)
      {
        size_it+=ref_sizes_[k];
        if(flag[k]==0)
        {


          int maxcov = -1;
          for(int i=0;i<ref_sizes_[k];i++)
          {
            if(output[k][i]>maxcov)
            {
              maxcov = output[k][i];
            }
          }
          if(maxcov > maxofmaxcov)
            maxofmaxcov = maxcov;
        }
      }


      long long int totalhistogram[maxofmaxcov+1];
      memset(totalhistogram,0,(maxofmaxcov+1)*sizeof(long long int));
      for(int k=0;k<ref_sizes_.size();k++)
      {



        if(flag[k]==0)
        {


          int maxcov = -1;
          for(int i=0;i<ref_sizes_[k];i++)
          {
            if(output[k][i]>maxcov)
            {
              maxcov = output[k][i];
            }
          }
          if(max_!=-1 && maxcov>max_)
            maxcov = max_;


          long long int histogram[maxcov+1];
          memset(histogram,0,(maxcov+1)*(sizeof(long long int)));
          for(int i=0;i<ref_sizes_[k];i++)
          {
            int val = output[k][i];
            if(val>maxcov)
              val = maxcov;
            histogram[val]= histogram[val]+(long long int)1;
          }

          for(int i=0;i<=maxcov;i++)
          {
            if(histogram[i]!=0)
            {
              totalhistogram[i]+=histogram[i];
              cout << ref_seqs_[k]<<"\t"<< i * scale_ <<"\t"<< histogram[i]   << "\t"<<ref_sizes_[k] << "\t" << (1.0 *  histogram[i])/(1.0 * ref_sizes_[k]) << endl ;
            }

            fflush(stdout);
          }

        }
        else
        {
          totalhistogram[0]+=ref_sizes_[k];
          cout << ref_seqs_[k]<<"\t"<< 0 * scale_ <<"\t"<< ref_sizes_[k]   << "\t"<<ref_sizes_[k] << "\t" << 1 << endl ;
        }

      }

      // output for a genome which contain many reference sequencies
      for(int i=0;i<=maxofmaxcov;i++)
      {
        if(totalhistogram[i]!=0)
        cout << "genome\t"<<i<<"\t"<<totalhistogram[i]<<"\t"<<size_it<<"\t"<<(1.0*totalhistogram[i])/(1.0*size_it)<<endl;
      }


    }



    ~AGDPeakDetectionOp()
    {
      if(dz_)
        print_res_d_or_dz(false);
      if(d_)
        print_res_d_or_dz(true);
      if(bg_)
        print_res_bg_or_bga(false);
      if(bga_)
        print_res_bg_or_bga(true);
      if(!d_ && !bg_ && !bga_ && !dz_)
        print_res_hist();


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


      char op;
      int op_len;
      int index = result->position().ref_index();

      if(flag[index]==-1)
      {

        output[index] = new int[ref_sizes_[index]];
        memset(output[index],0,ref_sizes_[index] * sizeof(int));
        flag[index]=0;
      }
      int start = result->position().position();



      while(cigar_len > 0)
      {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;

        if(op == 'M' || op=='X' || op=='=')  // TODO verify it with examples possibly different from standard
        {
          for(int i=0;i<op_len;i++)
          {
            if(strand_=="B" || (strand_=="+" && IsForwardStrand(flagf)) || (strand_=="-" && IsReverseStrand(flagf)))
            {
                output[index][start+i]++;
            }
          }

        }

        if(op=='D' || op=='M' || op=='N' || op=='X' || op=='=') // TODO verify it with examples possibly different from standard
          start+=op_len;

      }

      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {


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

      // done
      resource_releaser(results_container);
    }

  private:
    vector<string> ref_seqs_;
    vector<int32> ref_sizes_;
    // for various command line arguments
    int scale_;
    int **output;
    vector<int> flag;
    int max_;
    bool bg_;
    bool d_;
    bool dz_;
    string strand_;
    bool bga_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDPeakDetection").Device(DEVICE_CPU), AGDPeakDetectionOp);
} //  namespace tensorflow {
