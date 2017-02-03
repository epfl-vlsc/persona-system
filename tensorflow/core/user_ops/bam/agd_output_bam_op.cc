#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <sstream>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/agd-format/agd_record_reader.h"
#include "tensorflow/core/user_ops/agd-format/format.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/Bam.h"
#include "zlib.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  
  using shape_inference::InferenceContext;
   
  namespace { 
    void resource_releaser(ResourceContainer<Data> *data) {
      core::ScopedUnref a(data);
      data->release();
    }
  }

  REGISTER_OP("AgdOutputBam")
    .Attr("path: string")
    .Attr("pg_id: string")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_seq_sizes: list(int)")
    .Attr("read_group: string")
    .Attr("sort_order: {'unknown', 'unsorted', 'queryname', 'coordinate'}")
    .Input("results_handle: string")
    .Input("bases_handle: string")
    .Input("qualities_handle: string")
    .Input("metadata_handle: string")
    .Input("num_records: int32")
    .SetShapeFn([](InferenceContext* c) {
      return NoOutputs(c);
      })
    .SetIsStateful()
    .Doc(R"doc(
    On execution, append reads/results chunks to output BAM file. 

    Not all tags for SAM/BAM are currently supported, but support
    is planned. Currently supported is only required tags.

    RG and aux data is currently not supported but will be added soon.

    path: path for output .bam file
    pg_id: program id @PG for .bam
    ref_sequences: Reference sequences, @RG tags.
    ref_seq_sizes: Sizes of the references sequences.
    read_group: read group tag @RG
    *handles: the records to append to the BAM file
    num_records: the number of records held in *handles
    )doc");

  class AgdOutputBamOp : public OpKernel {
    public:
      explicit AgdOutputBamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
        string path, pg_id, read_group, sort_order;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &path));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pg_id", &pg_id));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("read_group", &read_group));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ref_seq_sizes", &ref_sizes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sort_order", &sort_order));

        OP_REQUIRES(ctx, ref_seqs_.size() == ref_sizes_.size(), 
            Internal("ref seqs was not same size as ref seq sizes lists"));
        stringstream header_ss;
        header_ss << "@HD VN:1.4 SO:";
        header_ss << sort_order;
        /*for (int i = 0; i < ref_seqs.size(); i++) {
          header_ss << "@SQ SN:" << ref_seqs[i] << " LN:" << to_string(ref_seq_sizes[i]) << endl;
        }*/
        ref_size_totals_.reserve(ref_seqs_.size());
        int64_t total = 0;
        for (int i = 0; i < ref_seqs_.size(); i++) {
          total += ref_sizes_[i];
          //ref_vec.push_back(RefData(ref_seqs_[i], ref_sizes_[i]));
          ref_size_totals_.push_back(total);
        }
        // open the file, we dont write yet
        bam_fp_ = fopen(path.c_str(), "w");
        string header = header_ss.str();

        scratch_.reset(new char[scratch_size_]); // 64K max size of BAM block
        scratch_compress_.reset(new char[scratch_size_]); // 64K max size of BAM block
        scratch_pos_ = 0;

        BAMHeader* bamHeader = (BAMHeader*) scratch_.get();
        bamHeader->magic = BAMHeader::BAM_MAGIC;
        size_t samHeaderSize = header.length();
        memcpy(bamHeader->text(), header.c_str(), samHeaderSize);
        bamHeader->l_text = (int)samHeaderSize;
        scratch_pos_ = BAMHeader::size((int)samHeaderSize);
		
        bamHeader->n_ref() = ref_seqs_.size();
		    BAMHeaderRefSeq* refseq = bamHeader->firstRefSeq();
        for (int i = 0; i < ref_seqs_.size(); i++) {
          int len = ref_seqs_[i].length() + 1;
          scratch_pos_ += BAMHeaderRefSeq::size(len);
          refseq->l_name = len;
          memcpy(refseq->name(), ref_seqs_[i].c_str(), len);
          refseq->l_ref() = (int)(ref_sizes_[i]);
          refseq = refseq->next();
          _ASSERT((char*) refseq - header == scratch_pos_);
        }

        // header is set in buffer, ready to add data
        zstream.zalloc = Z_NULL;
        zstream.zfree = Z_NULL;

      }

      ~AgdOutputBamOp() override {
        // if scratch_pos_ not 0, compress and append last bit and close file
        if (scratch_pos_ != 0) {
          CompressAndWrite();
        }
        // EOF marker for bam BGZF format
        static _uint8 eof[] = {
          0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43,
          0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
       
        // append the final EOF marker
        int status = fwrite(eof, sizeof(eof), 1, bam_fp_);
        
        if (status < 0) 
          LOG(INFO) << "WARNING: Final write of BAM eof marker failed with " << status;

        status = fclose(bam_fp_);
        
        if (status == EOF) 
          LOG(INFO) << "WARNING: Failed to close BAM file pointer: " << status;

      }

      void Compute(OpKernelContext* ctx) override {

        const Tensor *results_in, *bases_in, *qualities_in, *metadata_in, *num_records_t;
        OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
        OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_in));
        OP_REQUIRES_OK(ctx, ctx->input("bases_handle", &bases_in));
        OP_REQUIRES_OK(ctx, ctx->input("qualities_handle", &qualities_in));
        OP_REQUIRES_OK(ctx, ctx->input("metadata_handle", &metadata_in));
       
        auto num_records = num_records_t->scalar<int32>()();

        ResourceContainer<Data>* bases_data, *qual_data, *meta_data, *result_data;
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, bases_in, &bases_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, qualities_in, &qual_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, metadata_in, &meta_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, results_in, &result_data));

        AGDRecordReader base_reader(bases_data, num_records);
        AGDRecordReader qual_reader(qual_data, num_records);
        AGDRecordReader meta_reader(meta_data, num_records);
        AGDRecordReader results_reader(result_data, num_records);
       
        const format::AlignmentResult* result;
        const char* data, *meta, *base, *qual;
        const char* cigar;
        size_t len, meta_len, base_len, qual_len, cigar_len;
        int ref_index, mate_ref_index;
        vector<uint32_t> cigar_vec;
        cigar_vec.reserve(20); // should usually be enough

        Status s = results_reader.GetNextRecord(&data, &len);
        while (s.ok()) {
          OP_REQUIRES_OK(ctx, meta_reader.GetNextRecord(&meta, &meta_len));
          OP_REQUIRES_OK(ctx, base_reader.GetNextRecord(&base, &base_len));
          OP_REQUIRES_OK(ctx, qual_reader.GetNextRecord(&qual, &qual_len));
          const char* occ = strchr(meta, ' ');
          if (occ) 
            meta_len = occ - meta;
         
          result = reinterpret_cast<decltype(result)>(data);
          cigar = data + sizeof(format::AlignmentResult);
          cigar_len = len - sizeof(format::AlignmentResult);
          OP_REQUIRES_OK(ctx, ParseCigar(cigar, cigar_len, cigar_vec));
          
          size_t bamSize = BAMAlignment::size((unsigned)meta_len + 1, cigar_vec.size(), base_len, /*auxLen*/0);
          if ((scratch_size_ - scratch_pos_) < bamSize) {
            // compress and flush
            OP_REQUIRES_OK(ctx, CompressAndWrite());
          }
          
          BAMAlignment* bam = (BAMAlignment*) (scratch_.get() + scratch_pos_);
          bam->block_size = (int)bamSize - 4;

          int pos = FindChromosome(result->location_, mate_ref_index);
          bam->refID = ref_index;
          bam->pos = pos;
          bam->l_read_name = (_uint8)meta_len + 1;
          bam->MAPQ = result->mapq_;
          
          int mate_pos = FindChromosome(result->next_location_, ref_index);

          int refLength = cigar_vec.size() > 0 ? 0 : base_len;
          for (int i = 0; i < cigar_vec.size(); i++) {
              refLength += BAMAlignment::CigarCodeToRefBase[cigar_vec[i] & 0xf] * (cigar_vec[i] >> 4);
          }

          if (format::IsUnmapped(result)) {
            if (format::IsNextUnmapped(result)) {
              bam->bin = BAMAlignment::reg2bin(-1, 0);
            } else {
              bam->bin = BAMAlignment::reg2bin((int)mate_pos, (int)mate_pos+1);
            }
          } else {
            bam->bin = BAMAlignment::reg2bin((int)pos, (int)pos + refLength);
          }
            
          bam->n_cigar_op = cigar_vec.size();
          bam->FLAG = result->flag_;
          bam->l_seq = base_len;
          bam->next_refID = mate_ref_index;
          bam->next_pos = mate_pos;
          bam->tlen = (int)result->template_length_;
          memcpy(bam->read_name(), meta, meta_len);
          bam->read_name()[meta_len] = 0;
          memcpy(bam->cigar(), &cigar_vec[0], cigar_vec.size() * 4);
          BAMAlignment::encodeSeq(bam->seq(), base, base_len);
          memcpy(bam->qual(), qual, qual_len);
          for (unsigned i = 0; i < qual_len; i++) {
            bam->qual()[i] -= '!';
          }
          bam->validate();
         
          scratch_pos_ += bamSize;
        
          s = results_reader.GetNextRecord(&data, &len);
        }

        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        resource_releaser(result_data);

      }

    private:

      Status CompressAndWrite() {
        if (scratch_pos_ == 0)
          return Internal("attempting to compress and write 0 bytes");
        // set up BAM header structure
        gz_header header;
        _uint8 bamExtraData[6];
        header.text = false;
        header.time = 0;
        header.xflags = 0;
        header.os = 0;
        header.extra = bamExtraData;
        header.extra_len = 6;
        header.extra_max = 6;
        header.name = NULL;
        header.name_max = 0;
        header.comment = NULL;
        header.comm_max = 0;
        header.hcrc = false;
        header.done = true;
        bamExtraData[0] = 'B';
        bamExtraData[1] = 'C';
        bamExtraData[2] = 2;
        bamExtraData[3] = 0;
        bamExtraData[4] = 3; // will be filled in later
        bamExtraData[5] = 7; // will be filled in later

        const int windowBits = 15;
        const int GZIP_ENCODING = 16;
        zstream.next_in = (Bytef*) scratch_.get();
        zstream.avail_in = (uInt)scratch_pos_;
        zstream.next_out = (Bytef*) scratch_compress_.get();
        zstream.avail_out = (uInt)scratch_size_;
        uInt oldAvail;
        int status;

        status = deflateInit2(&zstream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowBits | GZIP_ENCODING, 8, Z_DEFAULT_STRATEGY);
        if (status < 0) 
          return Internal("libz deflate init failed with ", to_string(status));

        status = deflateSetHeader(&zstream, &header);
        if (status != Z_OK) {
          return Internal("libz: defaultSetHeader failed with", status);
        }

        oldAvail = zstream.avail_out;
        status = deflate(&zstream, Z_FINISH);

        if (status < 0 && status != Z_BUF_ERROR) {
          return Internal("libz: deflate failed with ", status);
        }
        if (zstream.avail_in != 0) {
          return Internal("libz: default failed to read all input");
        }
        if (zstream.avail_out == oldAvail) {
          return Internal("libz: default failed to write output");
        }
        status = deflateEnd(&zstream);
        if (status < 0) {
          return Internal("libz: deflateEnd failed with ", status);
        }

        size_t toUsed = scratch_size_ - zstream.avail_out;
        // backpatch compressed block size into gzip header
        if (toUsed >= BAM_BLOCK) {
          return Internal("exceeded BAM chunk size");
        }
        * (_uint16*) (scratch_compress_.get() + 16) = (_uint16) (toUsed - 1);

        status = fwrite(scratch_compress_.get(), toUsed, 1, bam_fp_);

        if (status < 0)
          return Internal("write compressed block to bam failed: ", status);

        scratch_pos_ = 0;

        return Status::OK();
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

      Status ParseCigar(const char* cigar, size_t cigar_len, vector<uint32_t>& cigar_vec) {
        // cigar parsing adapted from samblaster
        cigar_vec.clear();
        char op;
        int op_len;
        while(cigar_len > 0) {
          size_t len = parseNextOp(cigar, op, op_len);
          cigar += len;
          cigar_len -= len;         
          uint32_t val = (len << 4) | BAMAlignment::CigarToCode[op];
          cigar_vec.push_back(val);
        }
        return Status::OK();
      }

      int FindChromosome(int64_t location, int &ref_seq) {
         int index = 0;
         while (location > ref_size_totals_[index]) index++;
         ref_seq = index;
         return (index == 0) ? (int)location : (int)(location - ref_size_totals_[index-1]);
      }
    
      Status LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t, 
          ResourceContainer<Data>** container) {
        auto rmgr = ctx->resource_manager();
        auto handles_vec = handle_t->vec<string>();

        TF_RETURN_IF_ERROR(rmgr->Lookup(handles_vec(0), handles_vec(1), container));
        return Status::OK();
      }

      vector<string> ref_seqs_;
      vector<int32> ref_sizes_;
      vector<int64> ref_size_totals_;
      unique_ptr<char> scratch_, scratch_compress_;
      uint64_t scratch_pos_ = 0;
      static const uint64_t scratch_size_ = 64*1024; // 64Kb
      FILE* bam_fp_ = nullptr;
    
      z_stream zstream;
      

      TF_DISALLOW_COPY_AND_ASSIGN(AgdOutputBamOp);
  };


  REGISTER_KERNEL_BUILDER(Name("AgdOutputBam").Device(DEVICE_CPU), AgdOutputBamOp);

}  // namespace tensorflow
