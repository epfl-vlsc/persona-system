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
//usefull function
/* M= 0, I= 1, == 2, X= 3, S= 4, D= 5, N= 6, H= 7, P= 8, A= 9, T= 10, C= 11, G=12*/
      bool isOp(const char cigar){
        const char* text = "MI=XSDNHP";
        size_t size = 9;
        for (int i = 0 ; i < size ; i++){
          if(text[i] == cigar){
            return true;
          }
        }
        return false;
      }



//==============================================================================

    void Compute(OpKernelContext* ctx) override {

//==============================================================================
      if(bufferpair_pool_ == nullptr){
        OP_REQUIRES_OK(ctx,InitHandles(ctx));
      }

      const Tensor *compress_t, *chunk_size_t, *reference_t, *results_t;

      //ResourceContainer<Data> *reference_container;
      BasicContainer<AGDReferenceGenome> *reference_container;
      ResourceContainer<Data> *compress_container;
      ResourceContainer<Data> *results_container;
      ResourceContainer<BufferPair> *output_bufferpair_container;


      //TODO ok ca devrais être une fct qui retourne un statut. et qui prend le container en argument.
      OP_REQUIRES_OK(ctx, ctx->input("reference", &reference_t));
      OP_REQUIRES_OK(ctx, ctx->input("results", &results_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      OP_REQUIRES_OK(ctx, ctx->input("compress_base", &compress_t));
      OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx,&output_bufferpair_container));

      auto ref = reference_t->vec<string>();
      auto compress_base = compress_t->vec<string>();
      auto results = results_t->vec<string>();
      auto chunk_size = chunk_size_t->scalar<int32>()();
      auto output_bufferpair = output_bufferpair_container->get();
      auto rmgr = ctx->resource_manager();

      OP_REQUIRES_OK(ctx, rmgr->Lookup(ref(0),ref(1),&reference_container));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(compress_base(0), compress_base(1), &compress_container));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results(0), results(1), &results_container));

      ColumnBuilder column_builder;
      column_builder.SetBufferPair(output_bufferpair);

      //compress cigar
      const char* agd_compress;
      size_t compress_size;
      AGDRecordReader record_reader(compress_container,chunk_size);
      Status c = record_reader.GetNextRecord(&agd_compress,&compress_size);

      //metadata
      Alignment agd_result;
      AGDResultReader results_reader(results_container, chunk_size);
      Status s = results_reader.GetNextResult(agd_result);

      //reference
      AGDReferenceGenome *refGen = reference_container->get();

//usefull stuff for comparison =================================================
/* M= 0, I= 1, == 2, X= 3, S= 4, D= 5, N= 6, H= 7, P= 8 || A= 0, T= 1, C= 1, G=3*/
//initializes the needed val
      const char* oper = "MI=XSDNHP";
      const char* line = "|";
      size_t oper_size = 9;
      string val = "";
      int pos = 0;
      string tmp_genome = "";
      //copy the entire reference into decompress genome
      // for(int i = 0 ; i < reference_size; i++)|{
      //   decompress_genome += agd_reference[i];
      // }


      while(c.ok()){

        //ici il faut reinit le toute
        val = "";
        tmp_genome = "";
        pos = 0;

        const double position = agd_result.position().position();
        const double contig = agd_result.position().ref_index();
        //LOG(INFO) << "contig : "<< contig;
        if( contig > 93 || contig < 0){
          c = record_reader.GetNextRecord(&agd_compress,&compress_size);
          s = results_reader.GetNextResult(agd_result);
          continue;
        }
        const char* ref = refGen->GetSubstring(contig,position);

        const char* cigar = agd_result.cigar().c_str();
        // size_t cigar_len = agd_result.cigar().length();
        // const int flag = agd_result.flag();

        string refe = "";
        for(int i = 0 ; i < 101 ; i++){
          refe += ref[i];
        }

        // // recomposition de la string.
        string comp = "";
        for(int i = 0 ; i < compress_size ; i++){
          comp += agd_compress[i];
        }
        // LOG(INFO) << "here is the compression :" << comp;
        // LOG(INFO) << "here is the CIGAR : " << cigar;
        // LOG(INFO) << "CIGAR length : " << cigar_len;
        // LOG(INFO) << "results flag : " << flag;
        // LOG(INFO) << "results position : " << position;

        for(int i = 0; i < compress_size ; i++){
          // LOG(INFO) << "cigar : " << agd_compress[i];
          //if it's an op ======================================================
          if(isOp(agd_compress[i])){
            char op = agd_compress[i];
            // LOG(INFO) << "val = " << val;
            // LOG(INFO) << "op = " << op;
            int v = stoi(val);
            // LOG(INFO) << "val = " << v;
            //if it's a match ================================================
            if(op == oper[2]){
              // LOG(INFO) << "match !";
              for(int k = 0 ; k < v ; k++){
                //concatenate the matching bases
                tmp_genome += ref[pos+k];
              }
              if(agd_compress[i + 1] == line[0]){
                i++;
              }
              //update the genome position
              pos += v;
              //if it's a I or X ===============================================
            }else if(op == oper[1] || op == oper[3]){
              // LOG(INFO) << "v = " << v;
              // LOG(INFO) << "op = " << op;
              // LOG(INFO) << "its something changing from the ref !";
              for(int k = 0 ; k < v ; k++){
                if(agd_compress[i + 1] == line[0]){
                  i++;
                }
                //concatenate the matching bases
                tmp_genome += agd_compress[++i];
                // LOG(INFO) << "on a changer la base en : " << agd_compress[i];
                //update the genome position
                pos++;
              }
              //if it's a D ====================================================
            }else if(op == oper[5]){
              //update the genome position
              // LOG(INFO) << "val = " << val;
              // LOG(INFO) << "op = " << op;
              pos += stoi(val);
              if(agd_compress[i + 1] == line[0]){
                i++;
              }
              //if something else should not done so much ======================
            }else{
            }
            //reinit val
            val = "";
            //if it's a val ======================================================
          }else if(agd_compress[i] == line[0]){
            //do nothing
          }else{
            val += agd_compress[i];
          }
        }
        // LOG(INFO) << "reference = " << refe;
        // LOG(INFO) << "tmp genom = " << tmp_genome;
        c = record_reader.GetNextRecord(&agd_compress,&compress_size);
        s = results_reader.GetNextResult(agd_result);
        column_builder.AppendRecord(tmp_genome.c_str(),tmp_genome.length());
      }//while end
      LOG(INFO) << "genome decompressé";
      //release resource !
      //drfree(letter);
      resource_releaser(compress_container);
      resource_releaser(results_container);
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
