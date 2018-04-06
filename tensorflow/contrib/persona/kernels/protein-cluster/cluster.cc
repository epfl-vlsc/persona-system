
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
      auto& rep = seqs_[i];

      if (aligner.PassesThreshold(sequence.data, rep.Data(), sequence.length, rep.Length())) {

        LOG(INFO) << "passed threshold";
        ProteinAligner::Alignment alignment;
        // if subsequence homology, fully align and calculate coverages
        if (params->subsequence_homology) {
          if (sequence.total_seqs > rep.TotalSeqs() || (sequence.total_seqs == rep.TotalSeqs() 
              && *sequence.genome > rep.Genome()) || (*sequence.genome == rep.Genome() 
              && sequence.genome_index > rep.GenomeIndex()) ) {
      
            s = aligner.AlignLocal(rep.Data(), sequence.data, rep.Length(), sequence.length, alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq2_min, alignment.seq2_max);

          } else {
            s = aligner.AlignLocal(sequence.data, rep.Data(), sequence.length, rep.Length(), alignment);
            AddCoveredRange(*sequence.coverages, alignment.seq1_min, alignment.seq1_max);

          }
        }

        seqs_.push_back(ClusterSequence(string(sequence.data, sequence.length), *sequence.genome, 
              sequence.genome_index, sequence.total_seqs));
        return true;
      }
    }

    return false;

  }
}
