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
    }

    void Compute(OpKernelContext* ctx) override {

      // cout<<"called compute"<<endl;
      if (!input_queue_ && !bufpair_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }
      
      Tensor* out_t, *num_recs_t, *first_ord_t;

      if (unaligned_) {   // If required in the future
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({3, 2}), &out_t));
      } else {
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({4, 2}), &out_t));
      }

      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", TensorShape({}), &num_recs_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", TensorShape({}), &first_ord_t));
           
      auto& num_recs = num_recs_t->scalar<int>()();
      auto& first_ord = first_ord_t->scalar<int64>()();
      first_ord = first_ordinal_;

      ColumnBuilder base_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, base_builder, out_t, 0));
      ColumnBuilder qual_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, qual_builder, out_t, 1));
      ColumnBuilder meta_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, meta_builder, out_t, 2));
      AlignmentResultBuilder results_builder;
      OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, results_builder, out_t, 3));

      current_chunk_size = 0;
      Status s = Status::OK();
      Alignment result;
      const char *data_base,*data_qual,*data_meta;
      size_t len_base,len_qual,len_meta;

      // cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;

      if(!first_run)    //Finish reading records from previous chunk. (Except in first call of compute)
      {
        AGDRecordReader base_reader(bases_data, num_records);
        AGDRecordReader qual_reader(qual_data, num_records);
        AGDRecordReader meta_reader(meta_data, num_records);
        AGDResultReader results_reader(results_data, num_records);

        // cout<<"bringing pointer to last read \n";
        int i = 0;
        while(i < count_chunk_reads && s.ok())    // Getting to the position from where we have to continue reading
        {
            s = base_reader.GetNextRecord(&data_base,&len_base);
            s = qual_reader.GetNextRecord(&data_qual,&len_qual);
            s = meta_reader.GetNextRecord(&data_meta,&len_meta);
            s = results_reader.GetNextResult(result);
            i++;
        }

        // cout<<"resuming last chunk read now\n";

        s = base_reader.GetNextRecord(&data_base,&len_base);
        s = qual_reader.GetNextRecord(&data_qual,&len_qual);
        s = meta_reader.GetNextRecord(&data_meta,&len_meta);
        s = results_reader.GetNextResult(result);

        while( s.ok() && current_chunk_size < chunk_size_)
        {
          count_chunk_reads++;
          // cout<<"Scanning results of last chunk\n";
          if(ParseQuery(result))
          {
            // cout<<"Appending records\n";
            OP_REQUIRES_OK(ctx,IntoBases(data_base, len_base, bases_));
            base_builder.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
            qual_builder.AppendRecord(data_qual,len_qual);
            meta_builder.AppendRecord(data_meta,len_meta);
            results_builder.AppendAlignmentResult(result);
            current_chunk_size++;
          }
          // cout<<"Reading next record\n";
          s = base_reader.GetNextRecord(&data_base,&len_base);
          s = qual_reader.GetNextRecord(&data_qual,&len_qual);
          s = meta_reader.GetNextRecord(&data_meta,&len_meta);
          s = results_reader.GetNextResult(result);
          // cout<<"done reading\n";
        }

        // cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;
        // All records of last chunk have been analysed
        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        resource_releaser(results_data);        

      }

      first_run = false;

      Status dequeue_status;
      Status last_chunk_read = Status::OK();
      // cout<<"will dequeue new chunks from now\n";
      // cout<<"current_chunk_size : "<<current_chunk_size<<endl;

      while(current_chunk_size < chunk_size_)
      {
        // cout<<"about to dequeue\n";
        dequeue_status = DequeueElement(ctx);
        if(dequeue_status.ok())
        {
          // cout<<"ok deq status\n";
          AGDRecordReader base_reader(bases_data, num_records);
          AGDRecordReader qual_reader(qual_data, num_records);
          AGDRecordReader meta_reader(meta_data, num_records);
          AGDResultReader results_reader(results_data, num_records);

          count_chunk_reads = 0;
          s = Status::OK();
          while( s.ok() && current_chunk_size < chunk_size_)
          {
            // cout<<"current_chunk_size : "<<current_chunk_size;
            // cout<<"Reading next result\n";
            count_chunk_reads++;
            s = base_reader.GetNextRecord(&data_base,&len_base);
            s = qual_reader.GetNextRecord(&data_qual,&len_qual);
            s = meta_reader.GetNextRecord(&data_meta,&len_meta);
            s = results_reader.GetNextResult(result);
            // cout<<"Done reading next result..\n";
            if(ParseQuery(result) && s.ok())
            {
              // cout<<"passed filter, appending record\n";
              // cout<<"data_base "<<data_base<<"\nlen_base "<<len_base<<endl;
              // cout<<"data_qual "<<data_qual<<"\nlen_qual "<<len_qual<<endl;
              // cout<<"data_meta "<<data_meta<<"\nlen_meta "<<len_meta<<endl;
              OP_REQUIRES_OK(ctx,IntoBases(data_base, len_base, bases_));
              base_builder.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
              qual_builder.AppendRecord(data_qual,len_qual);
              meta_builder.AppendRecord(data_meta,len_meta);
              results_builder.AppendAlignmentResult(result);
              current_chunk_size++;
              // cout<<"Done appending\n";
            }
            else
            {
              // cout<<"didn't pass filter or s not ok\n";
            }
            if(!s.ok())   //chunk exhausted, no record read. (Counter had been incremented at the start of the loop)
              count_chunk_reads--;
          }
        // cout<<"count_chunk_reads : "<<count_chunk_reads<<endl;   //Should be equal to input chunk size
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

      cout<<"writing chunk with num_recs : "<<num_recs<<" and chunk size : "<<current_chunk_size<<endl;

      // cout<<"Done compute\n";
      OP_REQUIRES_OK(ctx,last_chunk_read);

    }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "bufpair_pool", &bufpair_pool_));
      
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
      input_queue_->TryDequeue(ctx, [&](const QueueInterface::Tuple &tuple) {
        // cout<<"Dequeuing\n";
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

          n.Notify();
          
        }

      });

      n.WaitForNotification();

      if(invalid == 2)
      {
        return Internal("End of file reached");
      }

      // cout<<"Done Dequeuing\n";
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

    bool ParseQuery(Alignment &result)
    {
      // TODO : Move these to constructor
      ANTLRInputStream input(predicate_);
      FilteringLexer lexer(&input);
      CommonTokenStream tokens(&lexer);

      // Loop to print the tokens if required
      // tokens.fill();
      // for (auto token : tokens.getTokens()) {
      //   cout << token->toString() << endl;
      // }

      FilteringParser parser(&tokens, result); 
      
      // parser.setBuildParseTree(false); // don't waste time bulding a tree
      // tree::ParseTree* tree = parser.prog(); // parse
      // cout << tree->toStringTree(&parser) << endl ; //To print the tree
      
      parser.prog();
      
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
    int32 num_records;
    ReferencePool<BufferPair> *bufpair_pool_ = nullptr;
    bool first_run;
    vector<BinaryBases> bases_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFiltering").Device(DEVICE_CPU), AGDFilteringOp);
} //  namespace tensorflow {
