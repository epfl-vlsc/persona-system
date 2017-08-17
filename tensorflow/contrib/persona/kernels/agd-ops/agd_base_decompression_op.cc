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
#include <algorithm>
#include <cctype>
#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class AGDBaseDecompressionOp : public OpKernel {
  public:
    AGDBaseDecompressionOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
      OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
      OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
      OP_REQUIRES_OK(context, context->GetAttr("ref_index", &ref_index));
      buffers_.resize(columns_.size());
      mmaps_.resize(columns_.size());
      ordinals_.resize(columns_.size());
      num_records_.resize(columns_.size());
      readers_.resize(columns_.size());
      fp = fopen("/scratch/bwa_index_hg19/hg19.fa", "r");
      if(fp == NULL)
        exit(EXIT_FAILURE);  
   
      for (int i = 0; i < 5; i++ ) {
         cout << ref_seqs_[i] << " index " << ref_index[i] << "\n";
      }
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
        string compressBase = "";
        for (int i = 0; i < columns_.size(); i++) {
          
	  if(columns_[i] == "base"){
	    OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            //fwrite(data, length, 1, stdout);
            base = data;
            compressBase = base.substr(0, length);
            cout << "compressed base: " << compressBase << "\n";
            //compressBase = 96=XC4=|99755945|6|16	  

	    string delimiter = "|";
            fseek(fp, 0, SEEK_SET);
            size_t pos = compressBase.find(delimiter);
            if (pos == -1) cout << "Wrong format, make sure the compressBase is in the correct format. Eg: 96=XC4=|99755945|6|16" << "\n";
            string modifiedCigar, location, contig, flag;
            modifiedCigar = compressBase.substr(0, pos);
            compressBase.erase(0, pos + delimiter.length());
            pos = compressBase.find(delimiter);
           // cout << "compress base now is" << compressBase;
            location = compressBase.substr(0, pos);
            compressBase.erase(0, pos + delimiter.length());
            pos = compressBase.find(delimiter);            
            contig = compressBase.substr(0, pos);
            compressBase.erase(0, pos + delimiter.length());
            pos = compressBase.find(delimiter);
            flag = compressBase.substr(0, pos);
           
            char * line = NULL;
            size_t linelen = 0;
            string decodedBase;
            while(getline(&line, &linelen , fp)){
               if (line[0] == '>') {
    		  string str(line);
		  str = str.substr(1,strlen(line) -2); 
                  if (str.compare(ref_seqs_[atoi(contig.c_str())]) == 0){
                   // cout << "length" << linelen;
                    pos = atoi(location.c_str());
                    while(pos > 50){
                      getline(&line, &linelen , fp);
                      pos = pos-50;
                    }
                    getline(&line, &linelen , fp);
                    string start(line);
                    start = start.substr(pos,strlen(line) - 1);
                    char currentR = start[0];
                    int lineChar = 0, cigarChar = 0;
                    string cigarRead = "";
                    int total = 0;
                    while (cigarChar != modifiedCigar.length()) {
                      if(currentR == '\n' || currentR == NULL){
                        getline(&line, &linelen , fp);
                        start.assign(line);
	                start = start.substr(0,strlen(line) - 1);
                        currentR = start[0];
                      }
                      if (modifiedCigar[cigarChar] == '=' || modifiedCigar[cigarChar] == 'D') {
                        bool skip = false;
                        if(modifiedCigar[cigarChar] == 'D') skip = true;
                        for(int j=0; j<atoi(cigarRead.c_str()); j++) {
                          if (!skip){
                            if ( flag.compare("16") == 0) {
                              if(toupper(currentR) == 'A') {decodedBase += 'T';}
                              else if(toupper(currentR) == 'C') {decodedBase += 'G';}
                              else if(toupper(currentR) == 'G') {decodedBase += 'C';}
                              else if(toupper(currentR) == 'T') {decodedBase += 'A';}
                              else {decodedBase += '?';}

                            }
                            else {
                              decodedBase += currentR;
                            }
                          }
        
                          lineChar++;
                          currentR = start[lineChar];
                          if(currentR == '\n' || currentR == NULL){
                            getline(&line, &linelen , fp);
                            start.assign(line);
                            start = start.substr(0,strlen(line) - 1);
                            currentR = start[0];
                            lineChar = 0;
                          }

                        }
                        cigarRead = "";
                        cigarChar++;
                      } else if (modifiedCigar[cigarChar] == 'X' || modifiedCigar[cigarChar] == 'I'){
                        bool skip = false;
                        if(modifiedCigar[cigarChar] == 'X') skip = true;
                        cigarChar++;
                        while(cigarChar != modifiedCigar.length() && !isdigit(modifiedCigar[cigarChar])) {
                           decodedBase += modifiedCigar[cigarChar];
                           if (skip) {
                             lineChar++;
                             currentR = start[lineChar];
                             if(currentR == '\n' || currentR == NULL){
                              getline(&line, &linelen , fp);
                              start.assign(line);
                              start = start.substr(0,strlen(line) - 1);
                              currentR = start[0];
                              lineChar = 0;
                            }
                           }
                           cigarChar++;
                        } 
                        cigarRead += modifiedCigar[cigarChar];
                        cigarChar++;
                      } else {
                        cigarRead += modifiedCigar[cigarChar];
                        cigarChar++;
                      }
                    }
                    for(auto& x: decodedBase)
                      x = toupper(x);
                    if ( flag.compare("16") == 0) 
                       reverse(decodedBase.begin(), decodedBase.end());
                    cout << "decoded" << decodedBase << "\n";
                    break;
                  }
               }
            }         
	} 
        }//for
        printf("\n");

        current++;
      } //while
      for (int i = 0; i < columns_.size(); i++) 
        mmaps_[i].reset(nullptr);

    }//compute

  private:
    vector<std::unique_ptr<ReadOnlyMemoryRegion>> mmaps_;

    //Buffer bases_buf_, qual_buf_, meta_buf_, results_buf_;
    vector<Buffer> buffers_;
    //uint64_t bases_ord_, qual_ord_, meta_ord_, results_ord_;
    vector<uint64_t> ordinals_;
    //uint32_t num_bases_, num_qual_, num_meta_, num_results_;
    vector<uint32_t> num_records_;
    vector<unique_ptr<AGDRecordReader>> readers_;
    FILE * fp;
    char * line;
    RecordParser rec_parser_;
    string record_id_;
    bool unpack_;
    vector<string> columns_;
    vector<string> ref_seqs_;
    vector<int> ref_index; 
  };//class

  REGISTER_KERNEL_BUILDER(Name("AGDBaseDecompression").Device(DEVICE_CPU), AGDBaseDecompressionOp);
} //  namespace tensorflow {
