
#include "tensorflow/contrib/persona/kernels/protein-cluster/cluster.h"

namespace tensorflow {

  using namespace std;

  bool Cluster::EvaluateSequence(const char* seq, int length, string& coverage, int num_reps, 
      const AlignmentEnvironments* envs, const Parameters* params) {
    ProteinAligner aligner(envs, params);
    Status s;

    for (size_t i = 0; i < num_reps && i < seqs_.size(); i++) {
      auto& rep = seqs_[i];

      if (aligner.PassesThreshold(seq, rep.data(), length, rep.length())) {

        // if subsequence homology, fully align and calculate coverages

        seqs_.push_back(Sequence(string(seq, length), 0));
        return true;
      }
    }

  }
}
