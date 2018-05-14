#pragma once

#include <vector>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"

namespace tensorflow {

// packaged sequence for evaluating against cluster
struct Sequence {
  const char* data;
  int length;
  std::string* coverages;
  std::string* genome;
  int genome_index;
  int total_seqs;
};

class Cluster {

  public:
    // create cluster and seed with sequence
    Cluster(const AlignmentEnvironments* envs, const char* seed_seq, int length, std::string genome,
        int genome_index, int total_seqs, int abs_seq) : envs_(envs), absolute_sequence_(abs_seq) {
      seqs_.push_back(ClusterSequence(string(seed_seq, length), genome, genome_index, total_seqs));
    }

    // return true if added to cluster, false if not
    // will modify coverage string (in `sequence`) if added
    bool EvaluateSequence(Sequence& sequence,  
        const AlignmentEnvironments* envs, const Parameters* params, 
        GenomeSequenceMap& candidate_map);

    int AbsoluteSequence() { return absolute_sequence_; }
    
    int TotalComps() { return total_comps_; }

    // BuildOutput() -- encode the seq pairs into tensors
    // RefinedMatches consist of 6 ints and 3 doubles
    // [idx1, idx2, score, distance, min1, max1, min2, max2, variance] 
    // [int, int, double, double, int, int, int, int, double]
    // which we split into an int tensor and a double tensor for downstream
    // transmission for aggregation and output. We need to use tensors so 
    // that we can transmit across machine boundaries. 
    //
    // Genomes (strings) are also required to disambiguate and find duplicates.
    //
    // [idx1, idx2, min1, max1, min2, max2], [score, distance, variance], [genome1, genome2]
    //
    // Shape([X, 6]), Shape([X, 3]), Shape([X,2])
    //
    // Where X is defined by whatever called BuildOutput()
    // Tensors need a defined Shape to be put into Queues
    //
    // Called after cluster has seen all sequences.
    Status BuildOutput(std::vector<Tensor>& match_ints, 
      std::vector<Tensor>& match_doubles, std::vector<Tensor>& match_genomes, 
      int size, OpKernelContext* ctx);

    size_t NumCandidates() { return candidates_.size(); }

  private:

    int total_comps_ = 0;
   
    // internal representation of sequences in the cluster
    class ClusterSequence {
      public:
        ClusterSequence(std::string seq, std::string genome, int idx, int total_seqs) : seq_(seq), 
          genome_(genome), genome_index_(idx), total_seqs_(total_seqs) {}

        const char* Data() const { return seq_.c_str(); }
        int Length() const { return int(seq_.length()); } 
        int GenomeIndex() const { return genome_index_; }
        const string& Genome() const { return genome_; }
        int TotalSeqs() const { return total_seqs_; }

      private:
        std::string seq_; // a copy, because we don't hang onto all chunks
        // these fields are primarily for constructing the output
        std::string genome_; // which dataset it belongs to
        int genome_index_; // what index in the dataset it was
        int total_seqs_; // total seqs in the genome this seq belongs to
    };

    struct Candidate {
      Candidate(int idx1, int idx2, const ProteinAligner::Alignment& alignment) :
        index_1(idx1), index_2(idx2), alignment(alignment) {}

      // indexes of sequences in seqs_ of this ClusterSequence
      int index_1;
      int index_2;

      ProteinAligner::Alignment alignment;
    };
    
    // perform all to all between sequences in the cluster
    // skip X first seqs, because they have been tested or added already
    void SeqToAll(const ClusterSequence* seq, int skip, ProteinAligner& aligner,
        GenomeSequenceMap& candidate_map); 

    bool PassesLengthConstraint(const ProteinAligner::Alignment& alignment,
        int seq1_len, int seq2_len);

    bool PassesScoreConstraint(const Parameters* params, int score);

    // representatives are just the first X seqs_
    std::vector<ClusterSequence> seqs_;
    // candidates are the result of intra cluster all to all
    std::vector<Candidate> candidates_;
    const AlignmentEnvironments* envs_; // for alignments
    int absolute_sequence_; // the absolute sequence of the chunk that
                            // produced this cluster
};

}
