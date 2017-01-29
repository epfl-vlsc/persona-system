#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <sstream>
#include "tensorflow/core/user_ops/bam/bamtools/src/api/BamWriter.h"
#include "tensorflow/core/user_ops/bam/bamtools/src/api/BamAlignment.h"
#include "tensorflow/core/user_ops/bam/bamtools/src/api/SamHeader.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/agd-format/agd_record_reader.h"
#include "tensorflow/core/user_ops/agd-format/format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace BamTools;
  
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

        stringstream header_ss;
        header_ss << "@HD VN:1.4 SO:";
        header_ss << sort_order << endl;
        header_ss << "@RG ID:" << read_group << endl;
        header_ss << "@PG ID:" << pg_id << endl;
        /*for (int i = 0; i < ref_seqs.size(); i++) {
          header_ss << "@SQ SN:" << ref_seqs[i] << " LN:" << to_string(ref_seq_sizes[i]) << endl;
        }*/
        RefVector ref_vec;
        ref_vec.reserve(ref_seqs_.size());
        ref_size_totals_.reserve(ref_seqs_.size());
        int64_t total = 0;
        for (int i = 0; i < ref_seqs_.size(); i++) {
          total += ref_sizes_[i];
          ref_vec.push_back(RefData(ref_seqs_[i], ref_sizes_[i]));
          ref_size_totals_.push_back(total);
        }

        SamHeader header(header_ss.str());
        OP_REQUIRES(ctx, bam_writer_.Open(path, header, ref_vec), Internal("AgdOutputBam: bamtools could not open file", path));
        
      }

      ~AgdOutputBamOp() override {
        bam_writer_.Close();
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
       
        BamAlignment alignment;
        const format::AlignmentResult* result;
        const char* data, *meta, *base, *qual;
        const char* cigar;
        size_t len, meta_len, base_len, qual_len, cigar_len;
        int ref_index;
        vector<CigarOp> cigar_vec;
        cigar_vec.reserve(20); // should usually be enough

        Status s = results_reader.GetNextRecord(&data, &len);
        while (s.ok()) {
          OP_REQUIRES_OK(ctx, meta_reader.GetNextRecord(&meta, &meta_len));
          OP_REQUIRES_OK(ctx, base_reader.GetNextRecord(&base, &base_len));
          OP_REQUIRES_OK(ctx, qual_reader.GetNextRecord(&qual, &qual_len));
          const char* occ = strchr(meta, ' ');
          if (occ) 
            meta_len = occ - meta;
          alignment.Name = string(meta, meta_len);
          alignment.QueryBases = string(base, base_len);
          alignment.Qualities = string(qual, qual_len);
         
          result = reinterpret_cast<decltype(result)>(data);
          cigar = data + sizeof(format::AlignmentResult);
          cigar_len = len - sizeof(format::AlignmentResult);
          alignment.AlignmentFlag = result->flag_;

          int pos = FindChromosome(result->location_, ref_index);
          alignment.RefID = ref_index;
          alignment.Position = pos;
          alignment.MapQuality = result->mapq_;
          OP_REQUIRES_OK(ctx, ParseCigar(cigar, cigar_len, cigar_vec));
          alignment.CigarData = cigar_vec;
          
          pos = FindChromosome(result->next_location_, ref_index);
          alignment.MateRefID = ref_index;
          alignment.MatePosition = pos;
          alignment.InsertSize = result->template_length_; // need to check if this is the same thing
          
          bam_writer_.SaveAlignment(alignment);
        
          s = results_reader.GetNextRecord(&data, &len);
        }

        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        resource_releaser(result_data);

      }

    private:

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

      Status ParseCigar(const char* cigar, size_t cigar_len, vector<CigarOp>& cigar_vec) {
        // cigar parsing adapted from samblaster
        cigar_vec.clear();
        char op;
        int op_len;
        while(cigar_len > 0) {
          size_t len = parseNextOp(cigar, op, op_len);
          cigar += len;
          cigar_len -= len;         
          cigar_vec.push_back(CigarOp(op, op_len));
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

      BamWriter bam_writer_;
      vector<string> ref_seqs_;
      vector<int32> ref_sizes_;
      vector<int64> ref_size_totals_;

      TF_DISALLOW_COPY_AND_ASSIGN(AgdOutputBamOp);
  };


  REGISTER_KERNEL_BUILDER(Name("AgdOutputBam").Device(DEVICE_CPU), AgdOutputBamOp);

}  // namespace tensorflow
