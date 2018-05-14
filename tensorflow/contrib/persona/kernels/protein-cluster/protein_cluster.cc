
#include "tensorflow/contrib/persona/kernels/protein-cluster/protein_cluster.h"

namespace tensorflow {

  using namespace std;

  void AddCoveredRange(string& coverages, size_t min, size_t max) {
    CHECK_LE(max, coverages.length());
    CHECK_LE(min, max);
    for (size_t i = min; i < max; i++)
      coverages[i] = 0;
  }

  bool Cluster::EvaluateSequence(Sequence& sequence,  
      const AlignmentEnvironments* envs, const Parameters* params, GenomeSequenceMap& candidate_map) {
    ProteinAligner aligner(envs, params);
    Status s;

    for (size_t i = 0; i < params->max_representatives && i < seqs_.size(); i++) {
      const auto& rep = seqs_[i];
      if (rep.Genome() == (*sequence.genome) && rep.GenomeIndex() == sequence.genome_index)
        break; // don't compare to itself, its already in this cluster

      total_comps_++;
      if (aligner.PassesThreshold(sequence.data, rep.Data(), sequence.length, rep.Length())) {

        //LOG(INFO) << "passed threshold";
        ProteinAligner::Alignment alignment;
        int skip = i;
        // if subsequence homology, fully align and calculate coverages
        if (params->subsequence_homology) {
          int index1, index2;
          skip++;
          if (sequence.total_seqs > rep.TotalSeqs() || (sequence.total_seqs == rep.TotalSeqs() 
              && *sequence.genome > rep.Genome()) || ((*sequence.genome == rep.Genome())
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

          /*if (PassesLengthConstraint(alignment, sequence.length, rep.Length()) &&
              PassesScoreConstraint(params, alignment.score)) {
            // add candidate for the good alignment with the representative
            LOG(INFO) << "adding candidate with alignment: " << alignment.score << ", " << alignment.pam_distance 
              << ", " << alignment.pam_variance;
            Candidate cand(index1, index2, alignment);
            candidates_.push_back(std::move(cand));
          }*/
        }

        ClusterSequence new_seq(string(sequence.data, sequence.length), *sequence.genome, 
            sequence.genome_index, sequence.total_seqs);

        SeqToAll(&new_seq, skip, aligner, candidate_map);

        seqs_.push_back(std::move(new_seq));
        return true;
      }
    }

    return false;

  }
    
  bool Cluster::PassesLengthConstraint(const ProteinAligner::Alignment& alignment,
    int seq1_len, int seq2_len) {

    float min_alignment_len = min(float(alignment.seq1_length), float(alignment.seq2_length));
    float max_min_seq_len = max(30.0f, 0.3f*float(min(seq1_len, seq2_len)));
    return min_alignment_len >= max_min_seq_len;
  }

  bool Cluster::PassesScoreConstraint(const Parameters* params, int score) {
    return score >= params->min_score;
  }
   
  void Cluster::SeqToAll(const ClusterSequence* seq, int skip, ProteinAligner& aligner, 
      GenomeSequenceMap& candidate_map) {

    const ClusterSequence* seq1 = seq;
    const ClusterSequence* seq2 = seq;
    int index1, index2;

    //LOG(INFO) << "SeqToAll against " << seqs_.size() - skip << " seqs in cluster";
    for (size_t i = 0; i < seqs_.size(); i++) {
      const auto* sequence = &seqs_[i];
      seq1 = seq; seq2 = seq;
      if (seq->TotalSeqs() > sequence->TotalSeqs() || 
          ((seq->TotalSeqs() == sequence->TotalSeqs()) && seq->Genome() > 
           sequence->Genome())) {
        seq1 = sequence; index1 = i;
        index2 = seqs_.size();
      } else {
        index1 = seqs_.size();
        seq2 = sequence; index2 = i;
      }
      
      if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() == seq2->GenomeIndex())
        continue;

      if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() > seq2->GenomeIndex()) {
        //LOG(INFO) << "seq 1 is greater than seq2 swapping ----------------------------";
        std::swap(seq1, seq2);
        std::swap(index1, index2);
        /*const auto& tmp = seq1;
        seq1 = seq2;
        seq2 = tmp;*/
      }
      
      auto genome_pair = make_pair(seq1->Genome(), seq2->Genome());
      auto seq_pair = make_pair(seq1->GenomeIndex(), seq2->GenomeIndex());
      auto genome_pair_it = candidate_map.find(genome_pair);
      if (genome_pair_it != candidate_map.end()) {
        auto seq_pair_it = genome_pair_it->second.find(seq_pair);
        if (seq_pair_it != genome_pair_it->second.end()) {
          LOG(INFO) << "not computing duplicate!";
          continue; // we already have this candidate
        }
      }


      total_comps_++;
      if (aligner.PassesThreshold(seq1->Data(), seq2->Data(), 
            seq1->Length(), seq2->Length())) {
           
        ProteinAligner::Alignment alignment;
        Status s = aligner.AlignLocal(seq1->Data(), seq2->Data(), 
            seq1->Length(), seq2->Length(), alignment);

        if (PassesLengthConstraint(alignment, seq1->Length(), seq2->Length()) &&
            PassesScoreConstraint(aligner.Params(), alignment.score)) {
          Candidate cand(index1, index2, alignment);
          candidates_.push_back(std::move(cand));
          candidate_map[genome_pair][seq_pair] = true;
        }

      } // else we don't add to candidates, score not high enough

    }
  }
    
  Status Cluster::BuildOutput(vector<Tensor>& match_ints, 
      vector<Tensor>& match_doubles, vector<Tensor>& match_genomes, 
      int size, OpKernelContext* ctx) {

    if (candidates_.size() == 0) return Status::OK();

    int num_tensors = candidates_.size() / size + 1;

    match_ints.resize(num_tensors);
    match_doubles.resize(num_tensors);
    match_genomes.resize(num_tensors);

    size_t cur_cand = 0;
    
    // [idx1, idx2, min1, max1, min2, max2], [score, distance, variance], [genome1, genome2]

    for (size_t i = 0; i < num_tensors; i++) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_INT32, TensorShape({size, 6}), &match_ints[i]));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_DOUBLE, TensorShape({size, 3}), &match_doubles[i]));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_STRING, TensorShape({size, 2}), &match_genomes[i]));

      auto ints = match_ints[i].matrix<int32>();
      auto doubles = match_doubles[i].matrix<double>();
      auto genomes = match_genomes[i].matrix<string>();

      for (size_t j = 0; j < size && cur_cand + j < candidates_.size(); j++) {
        const auto& candidate = candidates_[cur_cand + j];
        const auto& seq1 = seqs_[candidate.index_1];
        const auto& seq2 = seqs_[candidate.index_2];
        const auto& aln = candidate.alignment;

        ints(j, 0) = seq1.GenomeIndex();
        ints(j, 1) = seq2.GenomeIndex();
        ints(j, 2) = aln.seq1_min;
        ints(j, 3) = aln.seq1_max;
        ints(j, 4) = aln.seq2_min;
        ints(j, 5) = aln.seq2_max;
        doubles(j, 0) = aln.score;
        doubles(j, 1) = aln.pam_distance;
        doubles(j, 2) = aln.pam_variance;
        genomes(j, 0) = seq1.Genome();
        genomes(j, 1) = seq2.Genome();

        /*LOG(INFO) << "Candidate: " << seq1.Genome() << ", " << seq2.Genome() << " [" <<
          seq1.GenomeIndex()+1 << ", " << seq2.GenomeIndex()+1 << ", " << aln.score << ", " <<
          aln.pam_distance << ", " << aln.seq1_min+1 << ".." << aln.seq1_max+1 << ", " <<
          aln.seq2_min+1 << ".." << aln.seq2_max+1 << ", " << aln.pam_variance << "]";*/
      }
      cur_cand += size;
    }
    return Status::OK();
  }

}
