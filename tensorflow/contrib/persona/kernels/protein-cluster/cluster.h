#pragma once

#include <vector>
#include <string>
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"

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
        int genome_index, int total_seqs) : envs_(envs) {
      seqs_.push_back(ClusterSequence(string(seed_seq, length), genome, genome_index, total_seqs));
    }

    // return true if added to cluster, false if not
    // will modify coverage string if added
    bool EvaluateSequence(Sequence& sequence,  
        const AlignmentEnvironments* envs, const Parameters* params);

    // perform all to all between sequences in the cluster
    void AllToAll(); 

    // encode the seq pairs into tensors
    void BuildOutput();

  private:
   
    // internal representation of sequences in the cluster
    class ClusterSequence {
      public:
        ClusterSequence(std::string seq, std::string genome, int idx, int total_seqs) : seq_(seq), 
          genome_(genome), genome_index_(idx), total_seqs_(total_seqs) {}

        const char* Data() { return seq_.c_str(); }
        int Length() { return int(seq_.length()); } 
        int GenomeIndex() { return genome_index_; }
        string& Genome() { return genome_; }
        int TotalSeqs() { return total_seqs_; }

      private:
        std::string seq_; // a copy, because we don't hang onto all chunks
        // these fields are primarily for constructing the output
        std::string genome_; // which dataset it belongs to
        int genome_index_; // what index in the dataset it was
        int total_seqs_;
    };

    // representatives are just the first X seqs_
    std::vector<ClusterSequence> seqs_;
    const AlignmentEnvironments* envs_; // for alignments
};

}
