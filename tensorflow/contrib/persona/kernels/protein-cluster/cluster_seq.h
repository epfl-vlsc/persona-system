
#pragma once
#include <string>

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


}
