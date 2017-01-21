
#include "bwa_wrapper.h"
#include "bwa/bwamem.h"
#include "bwa/bntseq.h"

namespace bwa_wrapper {
  using namespace tensorflow;
  using namespace errors;
  using namespace std;
  
  static inline int get_rlen(int n_cigar, const uint32_t *cigar)
  {
    int k, l;
    for (k = l = 0; k < n_cigar; ++k) {
      int op = cigar[k]&0xf;
      if (op == 0 || op == 2)
        l += cigar[k]>>4;
    }
    return l;
  }

  Status BWAAligner::GenerateAndInfer(ReadResource* subchunk, mem_pestat_t pes[4]) {

    regs_.clear();
    const char* bases, *bases_mate;
    const char* quals, *quals_mate;
    size_t bases_len, mate_len;
    Status s = subchunk->get_next_record(&bases, &bases_len, &quals);
    regs_.reserve(subchunk->num_records());

    while (s.ok()) {
      s = subchunk->get_next_record(&bases_mate, &mate_len, &quals_mate);
      if (!s.ok())
        return Internal("subchunk was missing a read mate!");

      for (int i = 0; i < bases_len; ++i) // convert to 2-bit encoding 
        two_bit_seq_[i] = nst_nt4_table[(int)bases[i]];

      auto reg = mem_align1_core(options_, index_->bwt, index_->bns, index_->pac, bases_len, two_bit_seq_, nullptr);
      regs_.push_back(reg);

      for (int i = 0; i < mate_len; ++i) // convert to 2-bit encoding 
        two_bit_seq_[i] = nst_nt4_table[(int)bases_mate[i]];

      reg = mem_align1_core(options_, index_->bwt, index_->bns, index_->pac, mate_len, two_bit_seq_, nullptr);
      regs_.push_back(reg);

      s = subchunk->get_next_record(&bases, &bases_len, &quals);
    }

    if (!IsResourceExhausted(s))
      return s;

    // infer insert sizes a la BWAmem
    mem_pestat(options_, index_->bns->l_pac, subchunk->num_records(), &regs_[0], pes);

    if (!subchunk->reset_iter()) 
      return Internal("BWA aligner requires its read resource to be resettable");

    return Status::OK();
  }

  void BWAAligner::ProcessResult(mem_aln_t* bwaresult, mem_aln_t* bwamate, format::AlignmentResult& result, string& cigar) {

    result.flag_ = bwaresult->flag;
    result.flag_ |= bwamate ? 0x1 : 0; // is paired in sequencing
    result.flag_ |= bwaresult->rid < 0? 0x4 : 0; // is mapped
    result.flag_ |= bwamate && bwamate->rid < 0? 0x8 : 0; // is mate mapped
    if (bwaresult->rid < 0 && bwamate && bwamate->rid >= 0) { // copy mate to alignment
      result.location_ = bwamate->pos + index_->bns->anns[bwamate->rid].offset;
    } else
      result.location_ = bwaresult->pos + index_->bns->anns[bwaresult->rid].offset;

    result.flag_ |= bwaresult->is_rev? 0x10 : 0; // is on the reverse strand
    result.flag_ |= bwamate && bwamate->is_rev? 0x20 : 0; // is mate on the reverse strand
    result.next_location_ = 0;
    result.template_length_ = 0;
    result.mapq_ = bwaresult->mapq;
    cigar = "";

    int which = 0;
    if (bwaresult->rid >= 0) { // with coordinate
      if (bwaresult->n_cigar) { // aligned
        for (int i = 0; i < bwaresult->n_cigar; ++i) {
          int c = bwaresult->cigar[i]&0xf;
          if (!(options_->flag&MEM_F_SOFTCLIP) && !bwaresult->is_alt && (c == 3 || c == 4))
            c = which? 4 : 3; // use hard clipping for supplementary alignments
          cigar += to_string(bwaresult->cigar[i]>>4); 
          cigar += "MIDSH"[c];
        }
      } else cigar = "*"; // having a coordinate but unaligned (e.g. when copy_mate is true)
    } 

    if (bwamate && bwamate->rid >= 0) {

      result.next_location_ = bwamate->pos +(bwamate->is_rev? get_rlen(bwamate->n_cigar, bwamate->cigar) - 1 : 0);
      result.next_location_ += index_->bns->anns[bwamate->rid].offset;
      if (bwaresult->rid == bwamate->rid) {
        int64_t p0 = bwaresult->pos + (bwaresult->is_rev? get_rlen(bwaresult->n_cigar, bwaresult->cigar) - 1 : 0);
        int64_t p1 = bwamate->pos + (bwamate->is_rev? get_rlen(bwamate->n_cigar, bwamate->cigar) - 1 : 0);
        if (bwamate->n_cigar == 0 || bwaresult->n_cigar == 0) result.template_length_ = 0;
        else result.template_length_ = -(p0 - p1 + (p0 > p1? 1 : p0 < p1? -1 : 0));
      } else result.template_length_ = 0;
    } 
  }

  Status BWAAligner::AlignSubchunk(ReadResource *subchunk, AlignmentResultBuilder &result_builder) {
    // generate candidates

    mem_pestat_t pes[4];

    TF_RETURN_IF_ERROR(GenerateAndInfer(subchunk, pes));

    // to get abs position: bns->anns[p->rid] where p is mem_aln_t
    const char* bases, *bases_mate;
    const char* quals, *quals_mate;
    size_t bases_len, mate_len;
    Status s = subchunk->get_next_record(&bases, &bases_len, &quals);

    uint64_t id = 0; // num pairs
    size_t regs_index = 0;
    while (s.ok()) {
      s = subchunk->get_next_record(&bases_mate, &mate_len, &quals_mate);
      if (!s.ok())
        return Internal("subchunk was missing a read mate!");

      // BWA requires modifiable c-str buffers, so we have to copy :-(
      bseq1_t reads[2];
      reads[0].comment = 0; reads[1].comment = 0;
      reads[0].qual = strndup(quals, bases_len); reads[1].qual = strndup(quals_mate, mate_len);
      reads[0].seq = strndup(bases, bases_len); reads[1].seq = strndup(bases_mate, mate_len);
      reads[0].name = strdup(placeholder.c_str()); reads[1].name = strdup(placeholder.c_str());
      reads[0].l_seq = bases_len; reads[1].l_seq = mate_len;

      mem_aln_t results[2][2];
      int num_results[2];
      int ret = mem_sam_pe_results(options_, index_->bns, index_->pac, pes, id, reads, &regs_[regs_index], results, num_results);
      id++;
      regs_index += 2;

      format::AlignmentResult result, result_mate;
      string cigar, cigar_mate;
      ProcessResult(&results[0][0], &results[1][0], result, cigar);
      LOG(INFO) << "the absolute location is: " << result.location_;
      ProcessResult(&results[1][0], &results[2][0], result_mate, cigar_mate);
      LOG(INFO) << "the absolute location of mate is: " << result_mate.location_;

      result_builder.AppendAlignmentResult(result, cigar);
      result_builder.AppendAlignmentResult(result_mate, cigar_mate);

      s = subchunk->get_next_record(&bases, &bases_len, &quals);
    }

    if (!IsResourceExhausted(s))
      return s;

    return Status::OK();
  } 

} // namespace bwa wrapper
