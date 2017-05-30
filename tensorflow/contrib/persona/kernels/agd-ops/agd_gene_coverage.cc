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

  class AGDGeneCoverageOp : public OpKernel {
  public:
    AGDGeneCoverageOp(OpKernelConstruction *context) : OpKernel(context) {
        LOG(INFO) << "Started Finding Coverage " ;
        OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(context, context->GetAttr("ref_seq_sizes", &ref_sizes_));
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
        OP_REQUIRES_OK(context, context->GetAttr("bg", &bg_));
        OP_REQUIRES_OK(context, context->GetAttr("d", &d_));
        OP_REQUIRES_OK(context, context->GetAttr("strand", &strand_));
        OP_REQUIRES(context, ref_seqs_.size() == ref_sizes_.size(), Internal("ref seqs was not same size as ref seq sizes lists"));

        for(int i=0;i<ref_seqs_.size();i++)
        {
          flag.push_back(-1);
        }
        output = new int*[ref_seqs_.size()];
        //cout << ref_seqs_.size() << endl;
        fflush(stdout);








      // OP_REQUIRES_OK(context,context->GetAttr("ref_seq_sizes",&ref_sizes));
      // arr_size = 0;
      // for(int i=0;i<ref_sizes.size();i++)
      // {
      //   arr_size+=ref_sizes[i];
      // }
    }

    ~AGDGeneCoverageOp()
    {

      if(d_)
      {
        for(int i=0;i<ref_sizes_.size();i++)
        {
          if(flag[i]==0)
          {
            for(int j=0 ; j<ref_sizes_[i];j++)
            {
              if(output[i][j]!=0)
              {
                cout << ref_seqs_[i] << " " << j << " "<< output[i][j] << endl;
              }
            }
          }
        }
      }
      else
      {


      if(!bg_)
      {
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

            long long int histogram[maxcov+1];
            for(int i=0;i<=maxcov;i++)
            {
              histogram[i] = 0;
            }
            for(int i=0;i<ref_sizes_[k];i++)
            {
              int val = output[k][i];
              if(val>maxcov)
                val = maxcov;
              histogram[val]= histogram[val]+(long long int)1;
            }


            //int histogram[100];//max coverage 100X
            //cout << outputsize<<"total no of counters made"<< endl;

            for(int i=0;i<=maxcov;i++)
            {
              if(histogram[i]!=0)
              {            //cout << results_handle << " " << i<<" "<< histogram[i] << " "<<num_results << " " << (1.0 *  histogram[i])/(1.0 * num_results) << endl ;
                cout << ref_seqs_[k]<<" "<< i * scale_ <<" "<< histogram[i]   << " "<<ref_sizes_[k] << " " << (1.0 *  histogram[i])/(1.0 * ref_sizes_[k]) << endl ;
              }
              //cout << "No of base pairs with " << i << " coverage " << histogram[i]/scale<< endl ;
              fflush(stdout);
            }



          }

        }
      }
      else
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
                if(lastoutput!=0)
                {
                  cout << ref_seqs_[i]<<" "<< lastindex << " "<< j-1 << " "<< lastoutput * scale_ << endl;
                }
                lastindex = j ;
                lastoutput = output[i][j];
              }
            }
            if(lastoutput!=0)
            {
              cout << ref_seqs_[i]<<" "<< lastindex << " "<< ref_sizes_[i]-1 << " "<< lastoutput * scale_ << endl;
            }
          }
        }



      }
    }




        // int last_output = output[0];
        // int lastindex = 0;
        // for(int i=1;i<outputsize;i++)
        // {
        //   if(output[i]==last_output)
        //   {
        //     continue;
        //   }
        //   else
        //   {
        //     if(output[i-1]!=0)
        //       //cout << lastindex << " "<< i << " " << output[i-1]<< endl;
        //     lastindex = i+1;
        //     last_output = output[i+1];
        //   }
        // }
        // if(lastindex<=outputsize)
        // {
        //   cout << lastindex << " "<< outputsize << " "<<output[outputsize-1]<< endl;
        // }



      LOG(INFO) << "Done Finding Coverage " ;
    }



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

    Status CalculateCoverage(const Alignment *result,uint32 flagf) {
      // figure out the 5' position
      // the result->location is already genome relative, we shouldn't have to worry
      // about negative index after clipping, but double check anyway
      // cigar parsing adapted from samblaster
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      //cout << cigar << endl;

      char op;
      int op_len;
      int index = result->position().ref_index();
      if(flag[index]==-1)
      {
        output[index] = new int[ref_sizes_[index]];
        memset(output[index],0,sizeof(output[index]));
        for(int j=0;j<ref_sizes_[index];j++)
        {
          output[index][j]=0;
        }
        flag[index]=0;

      }
      int start = result->position().position();
      while(cigar_len > 0)
      {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;
        //LOG(INFO) << "cigar was " << op_len << " " << op;
        //LOG(INFO) << "cigar len is now: " << cigar_len;
        //cout << start <<"-"<<op_len << endl;
        // fflush(stdout);
        if (op == 'M')
        {
          for(int i=0;i<op_len;i++)
          {
            if(strand_=="B" || (strand_=="+" && IsForwardStrand(flagf)) || (strand_=="-" && IsReverseStrand(flagf)))
            { if(max_==-1)
                output[index][start+i]++;
              else if(max_!=-1 && output[index][start+i]<max_)
                output[index][start+i]++;
            }
          }
        }
        start+=op_len;
      }
      return Status::OK();
    }

    Status CalculateCoverage2(const Alignment *result,uint32 flagf)
    {
      // figure out the 5' position
      // the result->location is already genome relative, we shouldn't have to worry
      // about negative index after clipping, but double check anyway
      // cigar parsing adapted from samblaster
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      //cout << cigar << endl;

      char op;
      int op_len;
      int index = result->position().ref_index();
      if(flag[index]==-1)
      {
        output[index] = new int[ref_sizes_[index]];
        memset(output[index],0,sizeof(output[index]));
        for(int j=0;j<ref_sizes_[index];j++)
        {
          output[index][j]=0;
        }
        flag[index]=0;

      }
      int start = result->position().position();
      while(cigar_len > 0)
      {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;
        //LOG(INFO) << "cigar was " << op_len << " " << op;
        //LOG(INFO) << "cigar len is now: " << cigar_len;
        //cout << start <<"-"<<op_len << endl;
        // fflush(stdout);
        if (op == 'M')
        {
          for(int i=0;i<op_len;i++)
          {
            if(strand_=="B" || (strand_=="+" && IsForwardStrand(flagf)) || (strand_=="-" && IsReverseStrand(flagf)))
            { if(max_==-1)
                output[index][start+i]++;
              else if(max_!=-1 && output[index][start+i]<max_)
                output[index][start+i]++;
            }
          }
        }
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
      //cout << results_handle << " fg "<< num_results << endl;
      // get output buffer pairs (pair holds [index, data] to construct
      // the results builder for output
      // ResourceContainer<BufferPair> *output_bufferpair_container;
      // OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx, &output_bufferpair_container));
      // auto output_bufferpair = output_bufferpair_container->get();
      // AlignmentResultBuilder results_builder;
      // results_builder.SetBufferPair(output_bufferpair);
      LOG(INFO) << "reading result file " ;

      // Create an output tensor
      const Tensor& input_tensor = ctx->input(1);
      auto input = input_tensor.flat<int32>();

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto outputa = output_tensor->flat<int32>();

      // Set all but the first element of the output tensor to 0.
      const int N = input.size();
      for (int i = 0; i < N; i++) {
        outputa(i) = 0;
      }



    //auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.

      Alignment result;
      Status s = results_reader.GetNextResult(result);

      // this detection logic adapted from SamBlaster
      while (s.ok()) {
        if (!IsPrimary(result.flag()))
          OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
                result.position().DebugString()));

          // we have a single alignment
          if (IsUnmapped(result.flag())) {
            s = results_reader.GetNextResult(result);
            continue;
          }

          //LOG(INFO) << "processing mapped orphan at " << result->location_;
          OP_REQUIRES_OK(ctx, CalculateCoverage(&result,result.flag()));
        s = results_reader.GetNextResult(result);
        //cout << "reading a chunk of file" << endl;
        fflush(stdout);
      } // while s is ok()

      // done
      resource_releaser(results_container);





      //LOG(INFO) << "DONE running mark duplicates!! Found so far: " << num_dups_found_;

    }

  private:
    vector<string> ref_seqs_;
    vector<int32> ref_sizes_;
    int scale_;
    int **output;
    vector<int> flag;
    int max_;
    bool bg_;
    bool d_;
    string strand_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDGeneCoverage").Device(DEVICE_CPU), AGDGeneCoverageOp);
} //  namespace tensorflow {
