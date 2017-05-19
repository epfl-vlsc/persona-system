
#include "bwa_wrapper.h"
#include "bwa/bwamem.h"
#include "bwa/bntseq.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa/bwamem.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa/bntseq.h"

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

  Status BWAAligner::AlignSubchunkSingle(ReadResource* subchunk, vector<AlignmentResultBuilder>& result_builders) {

    const char* bases, *bases_mate;
    const char* quals, *quals_mate;
    size_t bases_len, mate_len;
    Alignment result, result_sup;
    string cigar, cigar_sup;

    Status s = subchunk->get_next_record(&bases, &bases_len, &quals);
    int64_t i = 0;
    bseq1_t read;
    mem_aln_t alignments[2];
    int num_alignments;
    while (s.ok()) {

      auto reg = mem_align1_core(options_, index_->bwt, index_->bns, index_->pac, bases_len, const_cast<char*>(bases), (void*)aux_);
		
      mem_mark_primary_se(options_, reg.n, reg.a, i);
		
      if (options_->flag & MEM_F_PRIMARY5) mem_reorder_primary5(options_->T, &reg);
      
      read.comment = 0; 
      read.qual = const_cast<char*>(quals); 
      // we use the remembered two bit seqs
      read.seq = const_cast<char*>(bases); 
      read.name = const_cast<char*>(placeholder.c_str()); 
      read.l_seq = bases_len;

      mem_reg2result(options_, index_->bns, index_->pac, &read, &reg, 0, 0, alignments, &num_alignments);
      
      // we only take the first result right now
      ProcessResult(&alignments[0], nullptr, result, cigar);
      result.set_cigar(cigar);
      result_builders[0].AppendAlignmentResult(result);
      if (num_alignments > 1) {
        LOG(INFO) << "result single has a supplemental";
        ProcessResult(&alignments[1], nullptr, result_sup, cigar_sup);
        result_sup.set_cigar(cigar_sup);
        result_builders[1].AppendAlignmentResult(result_sup);
      } else {
        LOG(INFO) << "appending an empty";
        result_builders[1].AppendEmpty();

      }
      //LOG(INFO) << "location " << result.location_ << " cigar: " << cigar;

      //TODO process XA tags of primary and append to secondary results columns
      
      free(reg.a);
      free(alignments[0].cigar);

      i++;
      s = subchunk->get_next_record(&bases, &bases_len, &quals);
    }

    if (!IsResourceExhausted(s))
      return s;

    return Status::OK();
  }

  Status BWAAligner::AlignSubchunk(ReadResource* subchunk, size_t index, vector<mem_alnreg_v>& regs) {

    const char* bases, *bases_mate;
    const char* quals, *quals_mate;
    size_t bases_len, mate_len;
    auto num_recs = subchunk->num_records();
    // this should only happen once
    /*if (two_bit_seqs_.size() < num_recs) {
      two_bit_seqs_.resize(num_recs);
      for (int i = 0; i < num_recs; i++) {
        two_bit_seqs_[i].resize(max_read_len_);
      }
    }*/

    Status s = subchunk->get_next_record(&bases, &bases_len, &quals);
    while (s.ok()) {
      s = subchunk->get_next_record(&bases_mate, &mate_len, &quals_mate);
      if (!s.ok())
        return Internal("subchunk was missing a read mate!");


      //LOG(INFO) << "bases len is " << bases_len;
      //memcpy(seq, bases, bases_len);
      /*for (int i  =0; i < bases_len; i++)
        if (seq[i] > 4)
          printf("seq is fucked\n");
        else
          printf("%u", (uint8_t)(seq[i]));
      printf("\n");*/
      auto reg = mem_align1_core(options_, index_->bwt, index_->bns, index_->pac, bases_len, const_cast<char*>(bases), (void*)aux_);
      regs[index++] = reg;


     // memcpy(seqmate, bases_mate, mate_len);
      reg = mem_align1_core(options_, index_->bwt, index_->bns, index_->pac, mate_len, const_cast<char*>(bases_mate), (void*)aux_);
      regs[index++] = reg;

      s = subchunk->get_next_record(&bases, &bases_len, &quals);
    }

    if (!IsResourceExhausted(s))
      return s;

    return Status::OK();
  }

  void BWAAligner::ProcessResult(mem_aln_t* bwaresult, mem_aln_t* bwamate, Alignment& result, string& cigar) {

    uint32 flag = bwaresult->flag;
    flag |= bwamate ? 0x1 : 0; // is paired in sequencing
    flag |= bwaresult->rid < 0? 0x4 : 0; // is mapped
    flag |= bwamate && bwamate->rid < 0? 0x8 : 0; // is mate mapped
    if (bwaresult->rid < 0 && bwamate && bwamate->rid >= 0) { // copy mate to alignment
      result.mutable_position()->set_position(bwamate->pos); // + index_->bns->anns[bwamate->rid].offset);
      result.mutable_position()->set_ref_index(bwamate->rid);
    } else {
      result.mutable_position()->set_position(bwaresult->pos); // + index_->bns->anns[bwamate->rid].offset);
      result.mutable_position()->set_ref_index(bwaresult->rid);
    }

    //LOG(INFO) << "location is: " << result.location_ - index_->bns->anns[bwaresult->rid].offset;
    flag |= bwaresult->is_rev? 0x10 : 0; // is on the reverse strand
    flag |= bwamate && bwamate->is_rev? 0x20 : 0; // is mate on the reverse strand
    result.set_flag(flag);
    //result.mutable_next_position()->set_position(0);
    //result.mutable_next_position()->set_ref_name("");
    result.set_template_length(0);
    result.set_mapping_quality(bwaresult->mapq);
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

    if (bwamate && bwamate->rid >= 0){
      //result.set_next_location(bwamate->pos + index_->bns->anns[bwamate->rid].offset);
      result.mutable_next_position()->set_position(bwamate->pos);
      result.mutable_next_position()->set_ref_index(bwamate->rid);
    }
    else if (bwamate) {
      //result.set_next_location(result.location()); // mate unmapped, set next to this ones location
      result.mutable_next_position()->set_position(result.position().position());
      result.mutable_next_position()->set_ref_index(result.position().ref_index());
    }
    else {
      result.mutable_next_position()->set_position(-1);
      result.mutable_next_position()->set_ref_index(-1);
    }

    if (bwamate && bwamate->rid >= 0) {

      if (bwaresult->rid == bwamate->rid) {
        int64_t p0 = bwaresult->pos + (bwaresult->is_rev? get_rlen(bwaresult->n_cigar, bwaresult->cigar) - 1 : 0);
        int64_t p1 = bwamate->pos + (bwamate->is_rev? get_rlen(bwamate->n_cigar, bwamate->cigar) - 1 : 0);
        if (bwamate->n_cigar == 0 || bwaresult->n_cigar == 0) result.set_template_length(0);
        else result.set_template_length(-(p0 - p1 + (p0 > p1? 1 : p0 < p1? -1 : 0)));
      } else result.set_template_length(0);
    } 
  }

  Status BWAAligner::FinalizeSubchunk(ReadResource *subchunk, size_t regs_index, vector<mem_alnreg_v>& regs, 
      mem_pestat_t pes[4], vector<AlignmentResultBuilder>& result_builders) {

    // to get abs position: bns->anns[p->rid] where p is mem_aln_t
    const char* bases, *bases_mate;
    const char* quals, *quals_mate;
    size_t bases_len, mate_len;
    Status s = subchunk->get_next_record(&bases, &bases_len, &quals);

    auto num_recs = subchunk->num_records();
    // this should only happen once
    /*if (two_bit_seqs_.size() < num_recs) {
      two_bit_seqs_.resize(num_recs);
      for (int i = 0; i < num_recs; i++) {
        two_bit_seqs_[i].resize(max_read_len_);
      }
    }*/
      
    bseq1_t reads[2];
    uint64_t id = 0; // num pairs
    while (s.ok()) {
      s = subchunk->get_next_record(&bases_mate, &mate_len, &quals_mate);
      if (!s.ok())
        return Internal("subchunk was missing a read mate!");

      // BWA requires modifiable c-str buffers, so we have to copy :-(
      //memcpy(seq, bases, bases_len);
      //memcpy(seqmate, bases_mate, mate_len);
      reads[0].comment = 0; reads[1].comment = 0;
      reads[0].qual = const_cast<char*>(quals); reads[1].qual = const_cast<char*>(quals_mate);
      // we use the remembered two bit seqs
      reads[0].seq = const_cast<char*>(bases); reads[1].seq = const_cast<char*>(bases_mate);
      reads[0].name = const_cast<char*>(placeholder.c_str()); reads[1].name = const_cast<char*>(placeholder.c_str());
      reads[0].l_seq = bases_len; reads[1].l_seq = mate_len;

      //LOG(INFO) << "regs index: " << regs_index;
      //LOG(INFO) << "read0: " << reads[0].seq << " : " << reads[0].qual << " : " << reads[0].l_seq;
      //LOG(INFO) << "read1: " << reads[1].seq << " : " << reads[1].qual << " : " << reads[1].l_seq;

      mem_aln_t results[2][2];
      int num_results[2];
      int ret = mem_sam_pe_results(options_, index_->bns, index_->pac, pes, id, reads, &regs[regs_index], results, num_results);
	
      // free data allocated by the align op
      // BWA really needs to stop allocating memory like this ...
      free(regs[regs_index|0].a); free(regs[regs_index|1].a);

      id++;
      regs_index += 2;

      Alignment result, result_mate, result_sup, result_sup_mate;
      string cigar, cigar_mate;
      ProcessResult(&results[0][0], &results[1][0], result, cigar);
      //LOG(INFO) << "cigar is: " << cigar;
      ProcessResult(&results[1][0], &results[0][0], result_mate, cigar_mate);
      //LOG(INFO) << "cigarmate is: " << cigar_mate;
      result.set_cigar(cigar);
      result_mate.set_cigar(cigar_mate);
      result_builders[0].AppendAlignmentResult(result);
      result_builders[0].AppendAlignmentResult(result_mate);
      if (num_results[0] > 1) {  // a supplemental for first 
        LOG(INFO) << "first had a supplemental!";
        ProcessResult(&results[0][1], nullptr, result_sup, cigar);
        result_sup.set_cigar(cigar);
        result_builders[1].AppendAlignmentResult(result_sup);
      } else
        result_builders[1].AppendEmpty();
      if (num_results[1] > 1) {  // a supplemental for second 
        LOG(INFO) << "second had a supplemental!";
        ProcessResult(&results[1][1], nullptr, result_sup_mate, cigar);
        result_sup_mate.set_cigar(cigar);
        result_builders[1].AppendAlignmentResult(result_sup_mate);
      } else
        result_builders[1].AppendEmpty();

      //TODO parse and process the XA tag data from the mem_aln_t structs to get secondary
      // alignments. 

      s = subchunk->get_next_record(&bases, &bases_len, &quals);
    }

    if (!IsResourceExhausted(s))
      return s;

    return Status::OK();
  } 

} // namespace bwa wrapper
