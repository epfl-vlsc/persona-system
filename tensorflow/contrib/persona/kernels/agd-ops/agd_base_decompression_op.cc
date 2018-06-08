#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <string>
#include <vector>
#include <iostream>
// #include <algorithm>
// #include <cctype>
// #include <stdio.h>
// #include <stdlib.h>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include "tensorflow/contrib/persona/kernels/agd-ops/agd_reference_genome.h"
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

  class AGDBaseDecompressionOp : public OpKernel {
  public:
    //Constructor
    AGDBaseDecompressionOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
//==============================================================================
    //   //is it really necessary ?
    //   OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
    //   OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
    //   OP_REQUIRES_OK(context, context->GetAttr("ref_index", &ref_index));
    //   buffers_.resize(columns_.size());
    //   mmaps_.resize(columns_.size());
    //   ordinals_.resize(columns_.size());
    //   num_records_.resize(columns_.size());
    //   readers_.resize(columns_.size());
    //
    //   fp = fopen("/scratch/bwa_index_hg19/hg19.fa", "r");
    //   if(fp == NULL)
    //     exit(EXIT_FAILURE);
    //
    //   for (int i = 0; i < 5; i++ ) {
    //      cout << ref_seqs_[i] << " index " << ref_index[i] << "\n";
    //   }
    // }
    //
    //
    // Status LoadChunk(OpKernelContext* ctx, string chunk_path) {
    //   //VLOG(INFO) << "chunk path is " << chunk_path;
    //   for (int i = 0; i < columns_.size(); i++) {
    //
    //     TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(
    //           chunk_path + "." + columns_[i], &mmaps_[i]));
    //     buffers_[i].reset();
    //     auto unpack = columns_[i] == "base" && unpack_;
    //     TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmaps_[i]->data(), mmaps_[i]->length(),
    //         true, &buffers_[i], &ordinals_[i], &num_records_[i], record_id_, unpack));
    //     readers_[i].reset(new AGDRecordReader(buffers_[i].data(), num_records_[i]));
    //   }
    //   return Status::OK();
//==============================================================================
    }
//==============================================================================
    //Added in accordance with the compression op.
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

      return Status::OK();
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("uncompress", ctx));
      return Status::OK();
    }

//==============================================================================

    void Compute(OpKernelContext* ctx) override {

//==============================================================================
      if(bufferpair_pool_ == nullptr){
        OP_REQUIRES_OK(ctx,InitHandles(ctx));
      }

      //TODO potentiellement que le reference container et tout ne sera pas necessaire
      //const Tensor *reference_t, *compress_t, *results_t, chunk_size_t;
      const Tensor *compress_t, *chunk_size_t, *reference_t;

      //ResourceContainer<Data> *reference_container;
      BasicContainer<AGDReferenceGenome> *reference_container;
      ResourceContainer<Data> *compress_container;
      //ResourceContainer<Data> *results_container;
      ResourceContainer<BufferPair> *output_bufferpair_container;

      OP_REQUIRES_OK(ctx,ctx->input("reference", &reference_t));
      //TF_RETURN_IF_ERROR(ctx->input("reference", &reference_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      OP_REQUIRES_OK(ctx, ctx->input("compress_base", &compress_t));
      OP_REQUIRES_OK(ctx,GetOutputBufferPair(ctx,&output_bufferpair_container));

      auto ref = reference_t->vec<string>();
      auto compress_base = compress_t->vec<string>();
      auto chunk_size = chunk_size_t->scalar<int32>()();
      auto output_bufferpair = output_bufferpair_container->get();
      auto rmgr = ctx->resource_manager();

      //TODO je doit changer la ligne ici pour pouvoir faire fonctionner dans l'idée
      //l'indexe fonctionne en fct de l'ordre des input et ici c'est le premier-
      //TF_RETURN_IF_ERROR(rmgr->Lookup(ref(0),ref(1),&reference_container));
      OP_REQUIRES_OK(ctx,rmgr->Lookup(ref(0),ref(1),&reference_container));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(compress_base(0), compress_base(1), &compress_container));

      ColumnBuilder column_builder;
      column_builder.SetBufferPair(output_bufferpair);

      //compress cigar
      Alignment agd_compress;
      AGDResultReader results_reader(compress_container, chunk_size);
      Status c = results_reader.GetNextResult(agd_compress);


      //reference
      AGDReferenceGenome *refGen = reference_container->get();

//usefull stuff for comparison =================================================
/* M = 0, I = 1, = = 2, X = 3, S = 4, D = 5, N = 6, H = 7, P= 8*/
      // LOG(INFO) << "Create the string to compare the CIGAR with";
      const char* text = "MI=XSDNHP";
      string val = "";
      string decompress_genome = "";
      char* letter;

      // int l = 0;
      // int v = 0;
      // letter = malloc(sizeof(char)*cigar_len)

      //copy the entire reference into decompress genome
      // for(int i = 0 ; i < reference_size; i++)|{
      //   decompress_genome += agd_reference[i];
      // }


      while(c.ok()){

        const char* cigar = agd_compress.cigar().c_str();
        size_t cigar_len = agd_compress.cigar().length();
        const int flag = agd_compress.flag();
        const double position = agd_compress.position().position();
        val = "";

        const char* ref = refGen->GetSubstring(agd_compress.position().ref_index(),position);
        for(int i = 0 ; i < 101 ; i++){
          LOG(INFO) << ref[i];
        }

        LOG(INFO) << "here is the CIGAR : " << cigar;
        LOG(INFO) << "CIGAR length : " << cigar_len;
        LOG(INFO) << "results flag : " << flag;
        LOG(INFO) << "results position : " << position;
        //TODO faire verifier ici je suis pas sur que la fonction soit .contig()
        // TODO faire comme dans compression et noté les algos au entre commentaire pour les tester
        //uint32_t contig = agd_compress.contig();
        //uint32_t pos = agd_compress.position().position();

        // val = ""
        // for(int i = 0; i < cigar_len; i++){
        //   if(cigar[i] is char quon connait){
        //     letter[l++] = cigar[i];
        //     if(!val.strcomp("")){
        //       //rajoute le chiffre dans le tab et remet le a 0
        //       val = ""
        //     }
        //   }else{
        //       val += cigar[i]
        //   }
        // }


        //modify the decompress genome according to the cigar.
        // for(int i = 0 ; i < cigar_len ; i++){
        //
        // }



        // c = record_reader.GetNextRecord(&agd_compress,&compress_size);
        // s = results_reader.GetNextResult(agd_result);
        // column_builder.AppendRecord(decompress_genome.c_str(),decompress_genome.length());
      }//while end

      //release resource !
      //drfree(letter);
      resource_releaser(compress_container);
//==============================================================================
    //   const Tensor *chunk_names_t, *start_t, *end_t, *chunk_size_t;
    //   OP_REQUIRES_OK(ctx, ctx->input("chunk_names", &chunk_names_t));
    //   OP_REQUIRES_OK(ctx, ctx->input("start", &start_t));
    //   OP_REQUIRES_OK(ctx, ctx->input("finish", &end_t));
    //   OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
    //   auto chunk_names = chunk_names_t->vec<string>();
    //   auto start = start_t->scalar<int>()();
    //   auto end = end_t->scalar<int>()();
    //   auto chunksize = chunk_size_t->scalar<int>()();
    //
    //   Status status;
    //   const char* data;
    //   size_t length;
    //   auto current = start;
    //   int which_chunk = current / chunksize;
    //
    //   OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
    //
    //   Alignment agd_result;
    //
    //   while (current <= end) {
    //     int chunk_offset = current - chunksize*which_chunk;
    //     if (chunk_offset >= chunksize) { // out of range, load next chunk
    //       which_chunk++;
    //       OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
    //       continue;
    //     }
    //     string base;
    //     string compressBase = "";
    //     for (int i = 0; i < columns_.size(); i++) {
    //
	  // if(columns_[i] == "base"){//or compressed base, however it is named
	  //   OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
    //         base = data;
    //         compressBase = base.substr(0, length);
    //         cout << "compressed base: " << compressBase << "\n";
    //
	  //   string delimiter = "|";
    //         //go to the beginning of the file
    //         fseek(fp, 0, SEEK_SET);
    //         size_t pos = compressBase.find(delimiter);
    //         if (pos == -1) cout << "Wrong format, make sure the compressBase is in the correct format. Eg: 96=XC4=|99755945|6|16" << "\n";
    //         //retrieve the modified cigar, location, contig (reference index), and flag (specifying the strand)
    //         string modifiedCigar, location, contig, flag;
    //         modifiedCigar = compressBase.substr(0, pos);
    //         compressBase.erase(0, pos + delimiter.length());
    //         pos = compressBase.find(delimiter);
    //         location = compressBase.substr(0, pos);
    //         compressBase.erase(0, pos + delimiter.length());
    //         pos = compressBase.find(delimiter);
    //         contig = compressBase.substr(0, pos);
    //         compressBase.erase(0, pos + delimiter.length());
    //         pos = compressBase.find(delimiter);
    //         flag = compressBase.substr(0, pos);
    //
    //         char * line = NULL;
    //         size_t linelen = 0;
    //         string decodedBase;
    //
    //         //loop through the file
    //         while(getline(&line, &linelen , fp)){
    //            // when the chromosome name is reached, check to see if that is equal to the reference contig specified above
    //            if (line[0] == '>') {
    // 		  string str(line);
		//   str = str.substr(1,strlen(line) -2);
    //               if (str.compare(ref_seqs_[atoi(contig.c_str())]) == 0) { //when the correct chromosome is reached
    //                 pos = atoi(location.c_str());
    //                 while(pos > 50){ //loop through the sequence until the location is reached
    //                   getline(&line, &linelen , fp);
    //                   pos = pos-50;
    //                 }
    //                 getline(&line, &linelen , fp);
    //                 string start(line);
    //                 start = start.substr(pos,strlen(line) - 1);
    //                 char currentR = start[0];
    //                 int lineChar = 0, cigarChar = 0;
    //                 string cigarRead = "";
    //                 int total = 0;
    //                 while (cigarChar != modifiedCigar.length()) { // once the position is reached, go through the cigar
    //                   if(currentR == '\n' || currentR == NULL){ // once at a new line continue to the next line until finishing reading the cigar
    //                     getline(&line, &linelen , fp);
    //                     start.assign(line);
	  //               start = start.substr(0,strlen(line) - 1);
    //                     currentR = start[0];
    //                   }
    //                   if (modifiedCigar[cigarChar] == '=' || modifiedCigar[cigarChar] == 'D') { // if a perfect match or deletion
    //                     bool skip = false;
    //                     if(modifiedCigar[cigarChar] == 'D') skip = true;
    //                     for(int j=0; j<atoi(cigarRead.c_str()); j++) {
    //                       if (!skip){ // if a perfect match
    //                         if ( flag.compare("16") == 0) {
    //                           if(toupper(currentR) == 'A') {decodedBase += 'T';}
    //                           else if(toupper(currentR) == 'C') {decodedBase += 'G';}
    //                           else if(toupper(currentR) == 'G') {decodedBase += 'C';}
    //                           else if(toupper(currentR) == 'T') {decodedBase += 'A';}
    //                           else {decodedBase += '?';}
    //
    //                         }
    //                         else {
    //                           decodedBase += currentR;
    //                         }
    //                       }
    //                       lineChar++;
    //                       currentR = start[lineChar];
    //                       if(currentR == '\n' || currentR == NULL){
    //                         getline(&line, &linelen , fp);
    //                         start.assign(line);
    //                         start = start.substr(0,strlen(line) - 1);
    //                         currentR = start[0];
    //                         lineChar = 0;
    //                       }
    //                     }
    //                     cigarRead = "";
    //                     cigarChar++;
    //                   } else if (modifiedCigar[cigarChar] == 'X' || modifiedCigar[cigarChar] == 'I'){ // if a mismatch or insertion
    //                     bool skip = false;
    //                     if(modifiedCigar[cigarChar] == 'X') skip = true;
    //                     cigarChar++;
    //                     while(cigarChar != modifiedCigar.length() && !isdigit(modifiedCigar[cigarChar])) {
    //                        decodedBase += modifiedCigar[cigarChar];
    //                        if (skip) { // if mismatch, skip the indices in the reference
    //                          lineChar++;
    //                          currentR = start[lineChar];
    //                          if(currentR == '\n' || currentR == NULL){
    //                           getline(&line, &linelen , fp);
    //                           start.assign(line);
    //                           start = start.substr(0,strlen(line) - 1);
    //                           currentR = start[0];
    //                           lineChar = 0;
    //                         }
    //                        }
    //                        cigarChar++;
    //                     }
    //                     cigarRead += modifiedCigar[cigarChar];
    //                     cigarChar++;
    //                   } else {
    //                     cigarRead += modifiedCigar[cigarChar];
    //                     cigarChar++;
    //                   }
    //                 }
    //                 for(auto& x: decodedBase) //convert to upper case
    //                   x = toupper(x);
    //                 if ( flag.compare("16") == 0) // if it's a negative strand, reverse the decoded sequence
    //                    reverse(decodedBase.begin(), decodedBase.end());
    //                 cout << "decoded" << decodedBase << "\n";
    //                 break;
    //               }
    //            }
    //         }
	  //        }
    //     }//for
    //     printf("\n");
    //     current++;
    //   } //while
    //   for (int i = 0; i < columns_.size(); i++)
    //     mmaps_[i].reset(nullptr);

//==============================================================================
    }//compute


//==============================================================================
  private:
    // vector<std::unique_ptr<ReadOnlyMemoryRegion>> mmaps_;
    // //Buffer bases_buf_, qual_buf_, meta_buf_, results_buf_;
    // vector<Buffer> buffers_;
    // //uint64_t bases_ord_, qual_ord_, meta_ord_, results_ord_;
    // vector<uint64_t> ordinals_;
    // //uint32_t num_bases_, num_qual_, num_meta_, num_results_;
    // vector<uint32_t> num_records_;
    // vector<unique_ptr<AGDRecordReader>> readers_;
    // FILE * fp;
    // char * line;
    // RecordParser rec_parser_;
    // vector<string> ref_seqs_;
    // vector<int> ref_index;

//==============================================================================
//ok
    const TensorShape scalar_shape_{};
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;
    string record_id_;
    bool unpack_;
    vector<string> columns_;
  };//class

  REGISTER_KERNEL_BUILDER(Name("AGDBaseDecompression").Device(DEVICE_CPU), AGDBaseDecompressionOp);
} //  namespace tensorflow {
