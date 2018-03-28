#pragma once

#include <vector>
#include <string>
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"

namespace tensorflow {

class Sequence {
  public:
    Sequence(std::string& seq, int id) : seq_(seq), genome_id_(id) {}
    
    const char* data() { return seq_.c_str(); }
    int length() { return int(seq_.length()); } 

  private:
    std::string seq_; // a copy, because we don't hang onto all chunks
    int genome_id_; 

};

class Cluster {

  public:
    Cluster(const AlignmentEnvironments* envs) : envs_(envs) {}

    // return true if added to cluster, false if not
    // will modify coverage string if added
    bool EvaluateSequence(const char* seq, int length, string& coverage);

    // perform all to all between sequences in the cluster
    void AllToAll(); 

    // encode the seq pairs into tensors
    void BuildOutput();

  private:
    std::vector<Sequence> seqs_;
    std::vector<Sequence*> reps_; // representatives
    const AlignmentEnvironments* envs_; // for alignments
};

}
