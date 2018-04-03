#pragma once

#include <vector>
#include <string>
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"

namespace tensorflow {

class Sequence {
  public:
    Sequence(std::string seq, int id) : seq_(seq), genome_id_(id) {}
    
    const char* data() { return seq_.c_str(); }
    int length() { return int(seq_.length()); } 

  private:
    std::string seq_; // a copy, because we don't hang onto all chunks
    // these fields are primarily for constructing the output
    int genome_id_; // which dataset it belongs to
    int genome_index_; // what index in the dataset it was

};

class Cluster {

  public:
    // create cluster and seed with sequence
    Cluster(const AlignmentEnvironments* envs, const char* seed_seq, int length) : envs_(envs) {
      seqs_.push_back(Sequence(string(seed_seq, length), 0));
    }

    // return true if added to cluster, false if not
    // will modify coverage string if added
    bool EvaluateSequence(const char* seq, int length, std::string& coverage, int num_reps, 
        const AlignmentEnvironments* envs, const Parameters* params);

    // perform all to all between sequences in the cluster
    void AllToAll(); 

    // encode the seq pairs into tensors
    void BuildOutput();

  private:
    // representatives are just the first X seqs_
    std::vector<Sequence> seqs_;
    const AlignmentEnvironments* envs_; // for alignments
};

}
