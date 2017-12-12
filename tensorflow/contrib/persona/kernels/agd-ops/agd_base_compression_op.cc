#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include <string>
#include <iostream>
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class AGDBaseCompressionOp : public OpKernel {
  public:
    AGDBaseCompressionOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
      OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
      buffers_.resize(columns_.size());
      mmaps_.resize(columns_.size());
      ordinals_.resize(columns_.size());
      num_records_.resize(columns_.size());
      readers_.resize(columns_.size());
    }

    Status LoadChunk(OpKernelContext* ctx, string chunk_path) {

      //VLOG(INFO) << "chunk path is " << chunk_path;
      for (int i = 0; i < columns_.size(); i++) {

        TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile( 
              chunk_path + "." + columns_[i], &mmaps_[i]));
        buffers_[i].reset();
        auto unpack = columns_[i] == "base" && unpack_;
        TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmaps_[i]->data(), mmaps_[i]->length(),
            true, &buffers_[i], &ordinals_[i], &num_records_[i], record_id_, unpack));
        readers_[i].reset(new AGDRecordReader(buffers_[i].data(), num_records_[i]));
      }
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor *chunk_names_t, *start_t, *end_t, *chunk_size_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_names", &chunk_names_t));
      OP_REQUIRES_OK(ctx, ctx->input("start", &start_t));
      OP_REQUIRES_OK(ctx, ctx->input("finish", &end_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      auto chunk_names = chunk_names_t->vec<string>();
      auto start = start_t->scalar<int>()();
      auto end = end_t->scalar<int>()();
      auto chunksize = chunk_size_t->scalar<int>()();

      Status status;
      const char* data;
      size_t length;
      auto current = start;
      int which_chunk = current / chunksize;

      OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));

      Alignment agd_result;

      while (current <= end) {
        int chunk_offset = current - chunksize*which_chunk;
        if (chunk_offset >= chunksize) { // out of range, load next chunk
          which_chunk++;
          OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
          continue;
        }
        string base;
        string strand =  to_string(agd_result.flag());
        string compressBase = "";
        for (int i = 0; i < columns_.size(); i++) {
          
	  if(columns_[i] == "base"){
	    OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            //fwrite(data, length, 1, stdout);
            base = data;
            base = base.substr(0, length);
            cout << "base: " << base << "\n";
	  } else if (columns_[i] == "results"){
            OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            agd_result.ParseFromArray(data, length);
            const char* cigar = agd_result.cigar().c_str();
	    size_t cigar_len = agd_result.cigar().length();
            cout << "cigar" << cigar << "\n";
            string read = "";
            int sum = 0;
            compressBase = "";
            for (int i=0; i < cigar_len; i++) {
               // if it's a mismatch or insertion
               if(cigar[i] == 'X' || cigar[i] == 'I') {
                 compressBase += cigar[i];
                 for(int j=0; j<atoi(read.c_str()); j++) {
                   if(strand.compare("16") == 0) {
                     compressBase += base.at(base.length()-1-sum);
                   } else {
                     compressBase += base.at(sum);
                   }
                   sum++; 
                 }
                 read = "";
               } else if (cigar[i] == '=' || cigar[i] == 'D') { //if a perfect match or deletion
                 compressBase += read;
                 compressBase += cigar[i];
                 if(cigar[i] == '=')
                    sum += atoi(read.c_str());
                 read = "";
               } else if (cigar[i] == 'M') {
                 LOG(INFO) << "Please realign using X and =";
               } else {
		 read += cigar[i];
               }
            }
            compressBase = compressBase + "|" + to_string(agd_result.position().position()) + "|" + to_string(agd_result.position().ref_index()) + "|" + to_string(agd_result.flag());
            cout <<"compressbase: " << compressBase << "\n";
          }
     
        }
        printf("\n");

        current++;
      }
      for (int i = 0; i < columns_.size(); i++) 
        mmaps_[i].reset(nullptr);

    }

  private:
    vector<std::unique_ptr<ReadOnlyMemoryRegion>> mmaps_;

    //Buffer bases_buf_, qual_buf_, meta_buf_, results_buf_;
    vector<Buffer> buffers_;
    //uint64_t bases_ord_, qual_ord_, meta_ord_, results_ord_;
    vector<uint64_t> ordinals_;
    //uint32_t num_bases_, num_qual_, num_meta_, num_results_;
    vector<uint32_t> num_records_;
    vector<unique_ptr<AGDRecordReader>> readers_;

    RecordParser rec_parser_;
    string record_id_;
    bool unpack_;
    vector<string> columns_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDBaseCompression").Device(DEVICE_CPU), AGDBaseCompressionOp);
} //  namespace tensorflow {
