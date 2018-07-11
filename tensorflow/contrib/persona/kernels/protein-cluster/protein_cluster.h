#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/multi_notification.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_executor.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/cluster_seq.h"

namespace tensorflow {

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
        const AlignmentEnvironments* envs, const Parameters* params);

    // return the absolute chunk sequence number that 'founded' this cluster
    int AbsoluteSequence() { return absolute_sequence_; }
   
    // call if on the fly SeqToAll is disabled
    void DoAllToAll(const AlignmentEnvironments* envs, 
      const Parameters* params);
   
    // for debug ----------------------
    int TotalComps() { return total_comps_; }

    int NumSequences() {return seqs_.size();}

    int LongestSeqLength();
    // -------------------------------

    // submit all to all alignments to the executor queue
    // using tuples of <seq1, seq2, alignment, notification>
    // returns number of alignments submitted
    int SubmitAlignments(AlignmentExecutor* executor, MultiNotification* n);

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
    // will wait on the cluster multi notification until all alignments are ready
    Status BuildOutput(std::vector<Tensor>& match_ints, 
      std::vector<Tensor>& match_doubles, std::vector<Tensor>& match_genomes, 
      int size, OpKernelContext* ctx);

    size_t NumCandidates() { return candidates_.size(); }

    void Dump(std::ostream& file);
    
  private:

    int total_comps_ = 0;
   

    struct Candidate {
      Candidate(const ClusterSequence* seq1, const ClusterSequence* seq2, 
          const ProteinAligner::Alignment& alignment) :
        seq_1(seq1), seq_2(seq2), alignment(alignment) {}

      // indexes of sequences in seqs_ of this ClusterSequence
      const ClusterSequence* seq_1;
      const ClusterSequence* seq_2;

      ProteinAligner::Alignment alignment;
    };
    
    bool PassesLengthConstraint(const ProteinAligner::Alignment& alignment,
        int seq1_len, int seq2_len);

    bool PassesScoreConstraint(const Parameters* params, int score);

    // representatives are just the first X seqs_
    std::list<ClusterSequence> seqs_;
    // candidates are the result of intra cluster all to all
    std::vector<Candidate> candidates_;
    const AlignmentEnvironments* envs_; // for alignments
    int absolute_sequence_; // the absolute sequence of the chunk that
                            // produced this cluster
    
    // alignments in order, for when the cluster is complete
    std::vector<ProteinAligner::Alignment> alignments_;
  
    mutex mu_;                    // protect modifications of seqs_
  
};

}
