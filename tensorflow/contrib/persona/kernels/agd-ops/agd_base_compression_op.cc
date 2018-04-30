#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <string>
#include <iostream>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
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
  }//namespace end

  class AGDBaseCompressionOp : public OpKernel {
  public:
    //Constructor
    AGDBaseCompressionOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
    }

    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

      return Status::OK();
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("ret", ctx));
      return Status::OK();
    }

    //our main function of compression.
    void Compute(OpKernelContext* ctx) override {
      if( bufferpair_pool_ == nullptr){
        OP_REQUIRES_OK(ctx,InitHandles(ctx));
      }

      // LOG(INFO) << "Starting compression";
      //create 2 tensor : 1 ressource and 1 chunk size
      const Tensor *results_t, *chunk_size_t, *records_t;

      // LOG(INFO) << "instanciate the container";
      ResourceContainer<Data> *results_container;
      ResourceContainer<Data> *records_container;
      ResourceContainer<BufferPair> *output_bufferpair_container;

      // LOG(INFO) << "initializes the tensor";
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      OP_REQUIRES_OK(ctx, ctx->input("results", &results_t));
      OP_REQUIRES_OK(ctx, ctx->input("records", &records_t));
      OP_REQUIRES_OK(ctx,GetOutputBufferPair(ctx,&output_bufferpair_container));

      // LOG(INFO) << "fill it";
      auto output_bufferpair = output_bufferpair_container->get();
      auto results = results_t->vec<string>();
      auto records = records_t->vec<string>();
      auto chunk_size = chunk_size_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();

      OP_REQUIRES_OK(ctx, rmgr->Lookup(records(0), records(1), &records_container));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results(0), results(1), &results_container));

      //for results.
      // LOG(INFO) << "instantiate the column builder for the output";
      ColumnBuilder column_builder;
      column_builder.SetBufferPair(output_bufferpair);

      //for cigar and other meta data.
      // LOG(INFO) << "instanciate the Status to get the CIGAR";
      Alignment agd_result;
      AGDResultReader results_reader(results_container, chunk_size);
      Status s = results_reader.GetNextResult(agd_result);

      //for data.
      // LOG(INFO) << "instantiate the Status to get the genome";
      const char* agd_record;
      size_t record_size;
      AGDRecordReader record_reader(records_container,chunk_size);
      Status p = record_reader.GetNextRecord(&agd_record,&record_size);

//usefull stuff for comparison =================================================
/* M = 0, I = 1, = = 2, X = 3, S = 4, D = 5, N = 6, H = 7, P= 8*/
      // LOG(INFO) << "Create the string to compare the CIGAR with";
      const char* text = "MI=XSDNHP";
      string val = "";
      int pos = 0;
      string compress_cigar = "";
//==============================================================================
      while(s.ok()){
        // LOG(INFO) << "CIGAR initialisation";
        const char* cigar = agd_result.cigar().c_str();
        size_t cigar_len = agd_result.cigar().length();
        const int flag = agd_result.flag();
        const double position = agd_result.position().position();

/*
use "|" as delimiter for the compress cigar to allow a better usage for decompress.
*/
        // LOG(INFO) << "analysing the CIGAR";
        for(int i = 0 ; i < cigar_len ; i++){
          char tmp = cigar[i];
          //LOG(INFO) << "here is the CIGAR : " << cigar[i];
          if(tmp == text[0]){
            LOG(INFO) << "should use X or =";
            ctx->SetStatus(errors::Internal("should use X or '=' instead of M"));
            return;
          }else if(tmp == text[1] || tmp == text[3] || tmp == text[5] ||tmp == text[6] ){
            pos += stoi(val);
            compress_cigar += val;
            compress_cigar += "|";
            compress_cigar += cigar[i];
            compress_cigar += "|";
            for(int j = 0; j < stoi(val) ; j++){
              compress_cigar += agd_record[pos+j];
            }
            compress_cigar += "|";
            val = "";
          }else if(tmp == text[2]){
            pos += stoi(val);
            compress_cigar += val;
            compress_cigar += "|";
            compress_cigar += cigar[i];
            compress_cigar += "|";
            val = "";
          }else if (tmp == text[7] ||tmp == text[8]  || tmp == text[4]){
            //TODO for moment do nothing but we should be trigger if we get this one time.
          }else{
            val += cigar[i];
          }
        }//for loop end

        pos = 0;
        // for(int i = 0 ; i < record_size; i++){
        //   LOG(INFO) << agd_record[i];
        // }
        LOG(INFO) << "record size : " << record_size;
        LOG(INFO) << "compress cigar : " << compress_cigar;
        LOG(INFO) << "here is the CIGAR : " << cigar;
        LOG(INFO) << "CIGAR length : " << cigar_len;
        LOG(INFO) << "results flag : " << flag;
        LOG(INFO) << "results position : " << position;

        p = record_reader.GetNextRecord(&agd_record,&record_size);
        s = results_reader.GetNextResult(agd_result);

        column_builder.AppendRecord(compress_cigar.c_str(),compress_cigar.length());
        //reset compress cigar
        compress_cigar = "";
      }//while s is ok()
      resource_releaser(results_container);
    }//compute end
  private:
    // RecordParser rec_parser_;
    const TensorShape scalar_shape_{};
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;
    string record_id_;
    bool unpack_;
    vector<string> columns_;
  };//class end


  REGISTER_KERNEL_BUILDER(Name("AGDBaseCompression").Device(DEVICE_CPU), AGDBaseCompressionOp);
} //  namespace tensorflow {
