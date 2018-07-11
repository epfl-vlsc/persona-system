      
#include "tensorflow/contrib/persona/kernels/protein-cluster/protein_cluster.h"

namespace tensorflow {

  using namespace std;

  void AddCoveredRange(string& coverages, size_t min, size_t max) {
    CHECK_LE(max, coverages.length());
    CHECK_LE(min, max);
    for (size_t i = min; i < max; i++)
      coverages[i] = 0;
  }
    
  void Cluster::Dump(ostream& file) {
    file << "{\n\"sequences\": [ \n";
    int index = 1;
    for (auto& seq : seqs_) {
      file << "{\n";
      file << "\"POSITION\": " << "\"" << seq.GenomeIndex() << "\",\n";
      file << "\"GENOME\": " << "\"" << seq.Genome() << "\"\n";
      if (index == seqs_.size()) {
        file << "}\n";
      } else {
        file << "},\n";
      }
      index++;
    }
    file << "]\n}";

  }

  bool Cluster::EvaluateSequence(Sequence& sequence,  
      const AlignmentEnvironments* envs, const Parameters* params) {
    ProteinAligner aligner(envs, params);
    Status s;

    auto seq_it = seqs_.begin();
    for (size_t i = 0; i < params->max_representatives && i < seqs_.size(); i++) {
      const auto& rep = *seq_it;
      seq_it++;
      if (rep.Genome() == (*sequence.genome) && rep.GenomeIndex() == sequence.genome_index)
        break; // don't compare to itself, its already in this cluster

      total_comps_++;
      //auto t1 = chrono::high_resolution_clock::now();
      bool passed = aligner.PassesThreshold(sequence.data, rep.Data(), sequence.length, rep.Length());
      // bool passed1 = aligner.PassesThresholdSSW(sequence.data, rep.Data(), sequence.length, rep.Length());
      //auto t2 = chrono::high_resolution_clock::now();
      //auto elapsed = chrono::duration_cast<chrono::microseconds>(t2 - t1);
      //outfile << sequence.length << ", " << rep.Length() << ", " << elapsed.count() << "\n";
      if (passed) {

        //LOG(INFO) << "passed threshold";
        ProteinAligner::Alignment alignment;
        int skip = i;
        // if subsequence homology, fully align and calculate coverages
        if (params->subsequence_homology) {
          skip++;
          if (sequence.total_seqs > rep.TotalSeqs() || (sequence.total_seqs == rep.TotalSeqs() 
              && *sequence.genome > rep.Genome()) || ((*sequence.genome == rep.Genome())
              && sequence.genome_index > rep.GenomeIndex()) ) {
      
            //s = aligner.AlignLocal(rep.Data(), sequence.data, rep.Length(), sequence.length, alignment);
            s = aligner.AlignSingle(rep.Data(), sequence.data, rep.Length(), sequence.length, alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq2_min, alignment.seq2_max);
          } else {

            //s = aligner.AlignLocal(sequence.data, rep.Data(), sequence.length, rep.Length(), alignment);
            s = aligner.AlignSingle(sequence.data, rep.Data(), sequence.length, rep.Length(), alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq1_min, alignment.seq1_max);
          }
        }

        ClusterSequence new_seq(string(sequence.data, sequence.length), *sequence.genome, 
            sequence.genome_index, sequence.total_seqs);

        {
          mutex_lock l(mu_);
          seqs_.push_back(std::move(new_seq));
        }
        return true;
      }
    }

    return false;

  }
    
  int Cluster::LongestSeqLength() {
    int len = 0;
    for (auto& seq : seqs_)
      if (seq.Length() > len)
        len = seq.Length();
    return len;
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
    
  int Cluster::SubmitAlignments(AlignmentExecutor* executor, MultiNotification* n) {

    int N = seqs_.size();
    //LOG(INFO) << N << " sequences";
    alignments_.resize(N*(N-1)/2);
    int aln = 0;
    for (auto it = seqs_.begin(); it != seqs_.end(); it++) {
      const auto* seq1 = &(*it);
      for (auto itj = next(it); itj != seqs_.end(); itj++) {
        //LOG(INFO) << "comparing seq " << i << " with seq j " << j;
        const auto* seq2 = &(*itj);
        auto item = make_tuple(seq1, seq2, &alignments_[aln], n);
        Status s = executor->EnqueueAlignment(item);
        if (!s.ok()) {
          LOG(INFO) << "error pushing!!! -----------------------";
          exit(0);
        }
        aln++;
      }
    }
    //LOG(INFO) << "submitted " << aln << ", " << N*(N-1)/2 << " alignments";
    CHECK_EQ(aln, N*(N-1)/2);
    return aln;
  }

  // post all-to-all
  void Cluster::DoAllToAll(const AlignmentEnvironments* envs, 
      const Parameters* params) {

    const ClusterSequence* seq1 = nullptr;
    const ClusterSequence* seq2 = nullptr;
    
    ProteinAligner aligner(envs, params);

    int aln = 0;
    //LOG(INFO) << "comparing " << seqs_.size() << " sequences";
    for (auto it = seqs_.begin(); it != seqs_.end(); it++) {
      const ClusterSequence* seq = &(*it);
      for (auto itj = next(it); itj != seqs_.end(); itj++) {
        //LOG(INFO) << "comparing seq " << i << " with seq j " << j;
        const auto* sequence = &(*itj);
        seq1 = seq; seq2 = seq;
        if (seq->TotalSeqs() > sequence->TotalSeqs() || 
            ((seq->TotalSeqs() == sequence->TotalSeqs()) && seq->Genome() > 
             sequence->Genome())) {
          seq1 = sequence;
        } else {
          seq2 = sequence;
        }

        if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() == seq2->GenomeIndex()) {
          aln++;
          continue;
        }

        if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() > seq2->GenomeIndex()) {
          std::swap(seq1, seq2);
        }

        // expects this to be filled by submitAlignments()
        auto& alignment = alignments_[aln];
        aln++;

        if (PassesLengthConstraint(alignment, seq1->Length(), seq2->Length()) &&
            PassesScoreConstraint(params, alignment.score)) {
          Candidate cand(seq1, seq2, alignment);
          candidates_.push_back(std::move(cand));
        }
      }
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
        const auto seq1 = candidate.seq_1;
        const auto seq2 = candidate.seq_2;
        const auto& aln = candidate.alignment;

        ints(j, 0) = seq1->GenomeIndex();
        ints(j, 1) = seq2->GenomeIndex();
        ints(j, 2) = aln.seq1_min;
        ints(j, 3) = aln.seq1_max;
        ints(j, 4) = aln.seq2_min;
        ints(j, 5) = aln.seq2_max;
        doubles(j, 0) = aln.score;
        doubles(j, 1) = aln.pam_distance;
        doubles(j, 2) = aln.pam_variance;
        genomes(j, 0) = seq1->Genome();
        genomes(j, 1) = seq2->Genome();

        /*LOG(INFO) << "Candidate: " << seq1->Genome() << ", " << seq2->Genome() << " [" <<
          seq1->GenomeIndex()+1 << ", " << seq2->GenomeIndex()+1 << ", " << aln.score << ", " <<
          aln.pam_distance << ", " << aln.seq1_min+1 << ".." << aln.seq1_max+1 << ", " <<
          aln.seq2_min+1 << ".." << aln.seq2_max+1 << ", " << aln.pam_variance << "]";*/
      }
      cur_cand += size;
    }
    return Status::OK();
  }

} // namespace tensorflow

