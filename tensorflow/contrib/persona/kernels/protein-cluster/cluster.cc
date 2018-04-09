
#include "tensorflow/contrib/persona/kernels/protein-cluster/cluster.h"

namespace tensorflow {

  using namespace std;

  void AddCoveredRange(string& coverages, size_t min, size_t max) {
    CHECK_LE(max, coverages.length());
    CHECK_LE(min, max);
    for (size_t i = min; i < max; i++)
      coverages[i] = 0;
  }

  bool Cluster::EvaluateSequence(Sequence& sequence,  
      const AlignmentEnvironments* envs, const Parameters* params) {
    ProteinAligner aligner(envs, params);
    Status s;

    for (size_t i = 0; i < params->max_representatives && i < seqs_.size(); i++) {
      const auto& rep = seqs_[i];

      if (aligner.PassesThreshold(sequence.data, rep.Data(), sequence.length, rep.Length())) {

        LOG(INFO) << "passed threshold";
        ProteinAligner::Alignment alignment;
        int skip = i;
        // if subsequence homology, fully align and calculate coverages
        if (params->subsequence_homology) {
          int index1, index2;
          skip++;
          if (sequence.total_seqs > rep.TotalSeqs() || (sequence.total_seqs == rep.TotalSeqs() 
              && *sequence.genome > rep.Genome()) || (*sequence.genome == rep.Genome() 
              && sequence.genome_index > rep.GenomeIndex()) ) {
      
            s = aligner.AlignLocal(rep.Data(), sequence.data, rep.Length(), sequence.length, alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq2_min, alignment.seq2_max);
            index1 = i;
            index2 = seqs_.size();

          } else {

            s = aligner.AlignLocal(sequence.data, rep.Data(), sequence.length, rep.Length(), alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq1_min, alignment.seq1_max);
            index2 = i;
            index1 = seqs_.size();
          }

          if (PassesLengthConstraint(alignment, sequence.length, rep.Length()) &&
              PassesScoreConstraint(params, alignment.score)) {
            // add candidate for the good alignment with the representative
            Candidate cand(index1, index2, alignment);
            candidates_.push_back(std::move(cand));
          }
        }

        ClusterSequence new_seq(string(sequence.data, sequence.length), *sequence.genome, 
            sequence.genome_index, sequence.total_seqs);

        SeqToAll(&new_seq, skip, aligner);

        seqs_.push_back(std::move(new_seq));
        return true;
      }
    }

    return false;

  }
    
  bool Cluster::PassesLengthConstraint(const ProteinAligner::Alignment& alignment,
    int seq1_len, int seq2_len) {

    auto min_alignment_len = min(alignment.seq1_length, alignment.seq2_length);
    auto max_min_seq_len = max(30, int(0.3f*min(seq1_len, seq2_len)));
    return min_alignment_len >= max_min_seq_len;
  }

  bool Cluster::PassesScoreConstraint(const Parameters* params, int score) {
    return score >= params->min_score;
  }
   
  void Cluster::SeqToAll(const ClusterSequence* seq, int skip, ProteinAligner& aligner) {

    const ClusterSequence* seq1 = seq;
    const ClusterSequence* seq2 = seq;
    int index1, index2;

    LOG(INFO) << "SeqToAll against " << seqs_.size() - skip << " seqs in cluster";
    for (size_t i = skip; i < seqs_.size(); i++) {
      const auto* sequence = &seqs_[i];
      if (seq->TotalSeqs() > sequence->TotalSeqs() || 
          (seq->TotalSeqs() == sequence->TotalSeqs() && seq->Genome() > sequence->Genome())) {
        seq1 = sequence; index1 = i;
        index2 = seqs_.size();
      } else {
        index1 = seqs_.size();
        seq2 = sequence; index2 = i;
      }

      if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() > seq2->GenomeIndex()) {
        std::swap(seq1, seq2);
        std::swap(index1, index2);
        /*const auto& tmp = seq1;
        seq1 = seq2;
        seq2 = tmp;*/
      }

      if (aligner.PassesThreshold(seq1->Data(), seq2->Data(), 
            seq1->Length(), seq2->Length())) {
           
        ProteinAligner::Alignment alignment;
        Status s = aligner.AlignLocal(seq1->Data(), seq2->Data(), 
            seq1->Length(), seq2->Length(), alignment);

        if (PassesLengthConstraint(alignment, seq1->Length(), seq2->Length()) &&
            PassesScoreConstraint(aligner.Params(), alignment.score)) {
          Candidate cand(index1, index2, alignment);
          candidates_.push_back(std::move(cand));
        }

      } // else we don't add to candidates, score not high enough

    }
  }

}
