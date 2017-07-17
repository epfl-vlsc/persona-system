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

#include "tensorflow/core/framework/queue_interface.h"

#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <sstream>
#include <tuple>
#include <thread>
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Bam.h"
#include "zlib.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"

#include <fstream>
#include "antlr4-runtime.h"
#include "FilteringLexer.h"
#include "FilteringParser.h"

using namespace antlr4;

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

  class AGDFilteringOp : public OpKernel {
  public:

    int64 count_chunk_reads;


    AGDFilteringOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      cout<<"Starting filtering constructor \n";
      count_chunk_reads = 0;
      stored_base_reader = nullptr;
      stored_qual_reader = nullptr;
      stored_meta_reader = nullptr;
      stored_results_reader = nullptr;
      first_run = true;

      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("unaligned", &unaligned_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("query", &predicate_));
    
      cout<<"chunk size : "<<chunk_size_<<endl;
      cout<<"query : "<<predicate_<<endl;

    }

    ~AGDFilteringOp() {
      core::ScopedUnref a1(input_queue_);
      core::ScopedUnref unref_listpool(bufpair_pool_);
      cout<<"Done filtering destructor \n";
      cout<<"Total : "<<count_chunk_reads<<endl;
    }

    void Compute(OpKernelContext* ctx) override {

      // const Tensor* results_t, *num_results_t;
      // OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      // OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      // auto results_handle = results_t->vec<string>();
      // auto num_results = num_results_t->scalar<int32>()();
      // auto rmgr = ctx->resource_manager();
      // ResourceContainer<Data> *results_container;
      // OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      // AGDResultReader results_reader(results_container, num_results);

      // Alignment result;
      // Alignment mate;
      // Status s = results_reader.GetNextResult(result);

      cout<<"called compute"<<endl;
      if (!input_queue_ && !bufpair_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }
      cout<<"Done init\n";
      fflush(stdout);
      // OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_t));
      // OP_REQUIRES_OK(ctx, ctx->GetAttr("unaligned", &unaligned_t));
      // OP_REQUIRES_OK(ctx, ctx->GetAttr("query", &predicate_t));
      
      // chunk_size_ = chunk_size_t->scalar<int>()();
      // unaligned_ = unaligned_t->scalar<bool>()();
      // predicate_ = predicate_t->scalar<string>()();

      Tensor* out_t, *num_recs_t, *first_ord_t;
      if (unaligned_) {
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({3, 2}), &out_t));
      } else {
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({4, 2}), &out_t));
      }

      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", TensorShape({}), &num_recs_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", TensorShape({}), &first_ord_t));
           
      cout<<"done allocating output\n";
      fflush(stdout);

      auto& num_recs = num_recs_t->scalar<int>()();
      auto& first_ord = first_ord_t->scalar<int64>()();
      first_ord = first_ordinal_;

      cout<<"starting column builder allocation\n";
      fflush(stdout);

      ColumnBuilder base_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, base_builder, out_t, 0));
      ColumnBuilder qual_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, qual_builder, out_t, 1));
      ColumnBuilder meta_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, meta_builder, out_t, 2));
      AlignmentResultBuilder results_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, results_builder, out_t, 3));

      cout<<"dONE column builder allocation\n";
      fflush(stdout);

      current_chunk_size = 0;
      Status s = Status::OK();
      Alignment result;
      const char *data_base,*data_qual,*data_meta;
      size_t len_base,len_qual,len_meta;

      cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;

      if(!first_run)
      {

      // srand(time(NULL));
          AGDRecordReader base_reader(bases_data, num_records);
          AGDRecordReader qual_reader(qual_data, num_records);
          AGDRecordReader meta_reader(meta_data, num_records);
          AGDResultReader results_reader(results_data, num_records);

          stored_base_reader = &base_reader;
          stored_qual_reader = &qual_reader;
          stored_meta_reader = &meta_reader;
          stored_results_reader = &results_reader;

      cout<<"bringing pointer to last read \n";
      int i = 0;
      while(i < count_chunk_reads && s.ok())
      {
          s = stored_base_reader->GetNextRecord(&data_base,&len_base);
          s = stored_qual_reader->GetNextRecord(&data_qual,&len_qual);
          s = stored_meta_reader->GetNextRecord(&data_meta,&len_meta);
          s = stored_results_reader->GetNextResult(result);
          i++;
      }

      cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;


      // if(stored_results_reader != nullptr)
      // {
        cout<<"resuming last chunk read\n";
        fflush(stdout);
        s = stored_base_reader->GetNextRecord(&data_base,&len_base);
        s = stored_qual_reader->GetNextRecord(&data_qual,&len_qual);
        s = stored_meta_reader->GetNextRecord(&data_meta,&len_meta);
              // cout<<"data_base ";
              // fflush(stdout);
              // cout<<data_base;
              // cout<<"\nlen_base "<<len_base<<endl;
              // cout<<"data_qual "<<data_qual<<"\nlen_qual "<<len_qual<<endl;
              // cout<<"data_meta "<<data_meta<<"\nlen_meta "<<len_meta<<endl;
              // fflush(stdout);
        if(s.ok())
          cout<<"ok\n";
        else
          cout<<"notok\n";
        s = stored_results_reader->GetNextResult(result);
        if(s.ok())
          cout<<"ok\n";
        else
          cout<<"notok\n";

        while( s.ok() && current_chunk_size < chunk_size_)
        {
          count_chunk_reads++;
          cout<<"Scanning results of last chunk\n";
          if(ParseQuery(result))
          {
            cout<<"Appending records\n";
            OP_REQUIRES_OK(ctx,IntoBases(data_base, len_base, bases_));
            base_builder.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
            // base_builder.AppendRecord(data_base,len_base);
            qual_builder.AppendRecord(data_qual,len_qual);
            meta_builder.AppendRecord(data_meta,len_meta);
            results_builder.AppendAlignmentResult(result);
            current_chunk_size++;
          }
          cout<<"Reading next record\n";
          s = stored_base_reader->GetNextRecord(&data_base,&len_base);
          s = stored_qual_reader->GetNextRecord(&data_qual,&len_qual);
          s = stored_meta_reader->GetNextRecord(&data_meta,&len_meta);
          // cout<<"data_base ";
          //     fflush(stdout);
          //     cout<<data_base;
          //     cout<<"\nlen_base "<<len_base<<endl;
          //     cout<<"data_qual "<<data_qual<<"\nlen_qual "<<len_qual<<endl;
          //     cout<<"data_meta "<<data_meta<<"\nlen_meta "<<len_meta<<endl;
          //     fflush(stdout);
          s = stored_results_reader->GetNextResult(result);
          cout<<"done reading\n";
        }

      cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;


        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        resource_releaser(results_data);        

      // }
      }
      first_run = false;

      Status dequeue_status;
      Status last_chunk_read = Status::OK();
      cout<<"will dequeue new chunks from now\n";
      fflush(stdout);
      cout<<"current_chunk_size : "<<current_chunk_size<<endl;

      while(current_chunk_size < chunk_size_)
      {
        cout<<"about to dequeue\n";
        fflush(stdout);
        dequeue_status = DequeueElement(ctx);
        cout<<"dequeued\n";
        if(dequeue_status.ok())
        {
          AGDRecordReader base_reader(bases_data, num_records);
          AGDRecordReader qual_reader(qual_data, num_records);
          AGDRecordReader meta_reader(meta_data, num_records);
          AGDResultReader results_reader(results_data, num_records);

          stored_base_reader = &base_reader;
          stored_qual_reader = &qual_reader;
          stored_meta_reader = &meta_reader;
          stored_results_reader = &results_reader;
          count_chunk_reads = 0;
          cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;
          fflush(stdout);
          cout<<"ok deq status\n";
          fflush(stdout);
          s = Status::OK();
          while( s.ok() && current_chunk_size < chunk_size_)
          {
            cout<<"current_chunk_size : "<<current_chunk_size;
            // cout<<"\nlen_base "<<len_base<<endl;
            cout<<"Reading next result\n";
            fflush(stdout);
            count_chunk_reads++;
            s = stored_base_reader->GetNextRecord(&data_base,&len_base);
            s = stored_qual_reader->GetNextRecord(&data_qual,&len_qual);
            s = stored_meta_reader->GetNextRecord(&data_meta,&len_meta);
            s = stored_results_reader->GetNextResult(result);
            cout<<"Done reading next result..\n";
            fflush(stdout);
            if(ParseQuery(result) && s.ok())
            {
              cout<<"passed filter, appending record\n";
              if(data_base == NULL)
              {
                // cout<<"data base is null\n";
                fflush(stdout);
              }
              // cout<<"jc\n";
              // cout<<"length is ";
              // fflush(stdout);
              // cout<<strlen(data_base);
              cout<<"data_base ";
              fflush(stdout);
              cout<<data_base;
              cout<<"\nlen_base "<<len_base<<endl;
              cout<<"data_qual "<<data_qual<<"\nlen_qual "<<len_qual<<endl;
              cout<<"data_meta "<<data_meta<<"\nlen_meta "<<len_meta<<endl;
              fflush(stdout);
              // base_builder.AppendRecord(data_base,len_base);
              OP_REQUIRES_OK(ctx,IntoBases(data_base, len_base, bases_));
              base_builder.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
              qual_builder.AppendRecord(data_qual,len_qual);
              meta_builder.AppendRecord(data_meta,len_meta);
              results_builder.AppendAlignmentResult(result);
              current_chunk_size++;
              cout<<"Done appending\n";
              fflush(stdout);
            }
            else
            {
              cout<<"didn't pass filter or s not ok\n";
            }
            cout<<"in loop after dequeuing and filtering\n";
            fflush(stdout);
            if(!s.ok())
              count_chunk_reads--;
          }
        cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;
        }
        else
        {
          //last chunk dequeued and filtered. Now exit and end compute
          cout<<"Last chunk dequeued\n";
          last_chunk_read = OutOfRange("No more chunks in dataset");
          break;
        }
      }

      num_recs = current_chunk_size;
      first_ordinal_ += current_chunk_size;
      cout<<"writing chunk with num_recs : "<<num_recs<<"and chunk size : "<<current_chunk_size<<endl;

      cout<<"Done compute\n";
      fflush(stdout);

      OP_REQUIRES_OK(ctx,last_chunk_read);

      // int i = 0;
      // cout<<"Dequeuing chunk "<<++i<<endl;
      // fflush(stdout);
      // Status s = DequeueElement(ctx);
      
      // while(s.ok())
      // {
      //   cout<<"Dequeuing chunk "<<++i<<endl;
      //   fflush(stdout);
      //   s = DequeueElement(ctx);
      //   cout<<"a"<<i<<"a";
      // }





      // Tensor *out_t;
      // OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({2}), &out_t));

      // // Grab the input tensor
      // const Tensor& input_tensor = ctx->input(1);
      // auto input = input_tensor.flat<int32>();
      // // Create an output tensor
      // Tensor* output_tensor = NULL;
      // OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),&output_tensor));
      // auto output = output_tensor->flat<int32>();
      // // cout<<"Filling values";
      // const int N = input.size();
      // for (int i = 0; i < N; i++) {
      //   output(i) = 0;
      // }
      // done
      // resource_releaser(results_container);
    }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "bufpair_pool", &bufpair_pool_));
      // auto &dtypes = input_queue_->component_dtypes();
      // if (dtypes.size() != 2) {
      //   return Internal("queue must have 2 elements, the first being the string id and latter being the desired id");
      // }
      // auto &record_id_type = dtypes[0];
      // auto &value_type = dtypes[1];
      // if (record_id_type != DT_STRING) {
      //   return Internal("first element of queue type must be string for record id");
      // }

      // if (value_type != dtype_) {
      //   return Internal("value type of queue does not match specified Batcher value type");
      // }

      return Status::OK();
    }

    Status LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t, ResourceContainer<Data>** container) {
      auto rmgr = ctx->resource_manager();
      auto handles_vec = handle_t->vec<string>();

      TF_RETURN_IF_ERROR(rmgr->Lookup(handles_vec(0), handles_vec(1), container));
      return Status::OK();
    }

    Status LoadDataResource(OpKernelContext* ctx, uint32 index, const Tensor* handle_t, ResourceContainer<Data>** container) {
      auto rmgr = ctx->resource_manager();
      auto handles_mat = handle_t->matrix<string>();

      TF_RETURN_IF_ERROR(rmgr->Lookup(handles_mat(index, 0), handles_mat(index, 1), container));
      return Status::OK();
    }

    Status DequeueElement(OpKernelContext *ctx) {
      Notification n;
      auto s = Status::OK();
      int invalid = -1;
      // cout<<"b";
      input_queue_->TryDequeue(ctx, [&](const QueueInterface::Tuple &tuple) {
        // auto &results_t = tuple[0];
        // auto &record_id = record_id_t.scalar<string>()();
        cout<<"Dequeuing\n";
        // cout<<tuple.size()<<endl;
        fflush(stdout);
        if(tuple.size() == 0)
        {
          invalid = 2;
          n.Notify();
        }
        else
        {
          num_records_t = &tuple[0];
          results_t = &tuple[1];
          bases_t = &tuple[2];
          quality_t = &tuple[3];
          metadata_t = &tuple[4];

          num_records = num_records_t->scalar<int32>()();

          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, bases_t, &bases_data));
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, quality_t, &qual_data));
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, metadata_t, &meta_data));
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, results_t, &results_data));


          // const char *data;
          // size_t len;
          // cout<<"len is "<<len<<endl;
          // fflush(stdout);
          // s = base_reader.GetNextRecord(&data,&len);
          // cout<<"len after reading is "<<len<<endl;
          // fflush(stdout);
          // s = stored_base_reader->GetNextRecord(&data,&len);
          // cout<<"len after second reading is "<<len<<endl;
          // fflush(stdout);


          // if (current_record_id_.empty()) {
          //   current_record_id_ = record_id;
          // } else if (record_id != current_record_id_) {
          //   s.Update(EnqueueOutput(ctx));
          //   if (s.ok()) {
          //     current_record_id_ = record_id;
          //     emitted = true;
          //   }
          // }

          // if (s.ok()) {
          //   batch_.push_back(move(tuple[1]));
          // }
          // emitted = false;
          
          // count_reads(&results_reader);
          // OP_REQUIRES_OK(ctx, count_reads(&results_reader));
          
          // resource_releaser(bases_data);
          // resource_releaser(qual_data);
          // resource_releaser(meta_data);
          // resource_releaser(results_data);
          
          n.Notify();
          
        }

      });

      n.WaitForNotification();

      if(invalid == 2)
      {
        return Internal("End of file reached");
      }

      // cout<<"Done Dequeuing\n";
      fflush(stdout);

      return s;
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      return Status::OK();
    }

    Status GetBufferForBuilder(OpKernelContext* ctx, ColumnBuilder& builder, Tensor* out, int index) {
      ResourceContainer<BufferPair> *output_bufpair_ctr;
      TF_RETURN_IF_ERROR(GetOutputBufferPair(ctx, &output_bufpair_ctr));
      auto output_bufferpair = output_bufpair_ctr->get();
      builder.SetBufferPair(output_bufferpair);
      auto out_mat = out->matrix<string>();
      out_mat(index, 0) = output_bufpair_ctr->container();
      out_mat(index, 1) = output_bufpair_ctr->name();
      return Status::OK();
    }

    Status count_reads(AGDResultReader* results_reader)
    {
      Alignment result;
      // cout<<"q";
      fflush(stdout);
      Status s = results_reader->GetNextResult(result);
      while (s.ok()) {
        count_chunk_reads++;
        // cout<<"Reads till now : "<<count_chunk_reads<<"\n";
        s = results_reader->GetNextResult(result);
      } // while s is ok()
      return Status::OK();
    }

    bool RandomQueryResult()
    {
      // int n = (++random)%2;
      // if(n==0) return false;
      // else return true;
      return true;
    }

    bool ParseQuery(Alignment &result)
    {
      ANTLRInputStream input(predicate_);
      FilteringLexer lexer(&input);
      CommonTokenStream tokens(&lexer);

      // TODO : Move these to constructor
      
      // tokens.fill();
      // for (auto token : tokens.getTokens()) {
      //   cout << token->toString() << endl;
      // }

      FilteringParser parser(&tokens, result); 
      
      // parser.setBuildParseTree(false); // don't waste time bulding a tree
      
      // tree::ParseTree* tree = parser.prog(); // parse

      parser.prog();
      
      // cout << tree->toStringTree(&parser) << endl ;

      return parser.answer;
    }

  private:
    QueueInterface *input_queue_ = nullptr;
    const Tensor *chunk_size_t,*unaligned_t,*predicate_t;
    const Tensor *results_t,*bases_t,*quality_t,*metadata_t,*num_records_t;
    ResourceContainer<Data> *bases_data, *qual_data, *meta_data, *results_data;
    int chunk_size_;
    bool unaligned_;
    string predicate_;
    int current_chunk_size;
    int64 first_ordinal_ = 0;
    AGDRecordReader *stored_base_reader, *stored_qual_reader, *stored_meta_reader;
    AGDResultReader *stored_results_reader;
    int32 num_records;
    ReferencePool<BufferPair> *bufpair_pool_ = nullptr;
    int random = 0;
    bool first_run;
    vector<BinaryBases> bases_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFiltering").Device(DEVICE_CPU), AGDFilteringOp);
} //  namespace tensorflow {
