
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class AGDClusterAggregateOp : public OpKernel {
  public:
    AGDClusterAggregateOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("output_dir", &output_dir_));
      auto env = context->env();
      env_ = env; // i think this is safe?
     
      OP_REQUIRES_OK(context, env->IsDirectory(output_dir_));
      
    }

    ~AGDClusterAggregateOp() {
      // dump output to disk somewhere
      // one folder per genome, one file per pair in each folder

      Status s;
      size_t total_candidates = 0;
      for (auto& matches_kv : all_matches_map_) {
        auto& genome_pair = matches_kv.first;
        auto& candidate_map = matches_kv.second;
        string path = output_dir_ + "/" + genome_pair.first;
        LOG(INFO) << "creating dir: " << path;
        LOG(INFO) << "out dir : " << output_dir_ << " pair first: " << genome_pair.first;
        s= env_->CreateDir(path);
        if (!IsAlreadyExists(s) && !s.ok()) {
          LOG(INFO) << "not able to create dir: " << path << ", aborting output.";
          return;
        }
        path += string("/") + genome_pair.second;
        LOG(INFO) << "creating file: " << path;
        std::unique_ptr<WritableFile> outfile;
        s = env_->NewWritableFile(path, &outfile);
        if (!s.ok()) {
          LOG(INFO) << "not able to create file: " << path << ", aborting output.";
          return;
        }

        if (candidate_map.size() > 0) {
          s = outfile->Append(string("# AllAll of ") + genome_pair.first + " vs " +
              genome_pair.second + ";\nRefinedMatches(\n[");
          if (!s.ok()) {
            LOG(INFO) << "not able to append to: " << path << ", aborting output.";
            return;
          }
        } else {
          s = outfile->Append(string("# AllAll of ") + genome_pair.first + " vs " +
              genome_pair.second + ";\nRefinedMatches(\n[] ):");
          if (!s.ok()) {
            LOG(INFO) << "not able to append to: " << path << ", aborting output.";
            return;
          }
          return;
        }

        for (auto it(candidate_map.begin()); it != candidate_map.end(); it++) {
        
          total_candidates++;
          auto& match = it->second;
          ostringstream ss;
          ss << "[" << match.index_1 + 1 << ", " << match.index_2 + 1 << ", ";
          ss << std::fixed;
          ss.precision(7);
          ss << round(match.score * 10000000.0f) / 10000000.0f << ", ";
          if (match.distance >= 45.0f)
            ss << int(match.distance);
          else if (match.distance > 0.1f) {
            ss.precision(4);
            ss << round(match.distance * 10000.0f) / 10000.0f;
          } else {
            ss.precision(8);
            ss << round(match.distance * 100000000.0f) / 100000000.0f;
          }
          ss.precision(8);

          ss << ", " << match.seq1_min + 1 << ".." << match.seq1_max + 1
            << ", " << match.seq2_min + 1 << ".." << match.seq2_max + 1 << ", " 
            << round(match.variance * 100000000.0f) / 100000000.0f
            << "]";

          if (std::distance(it, candidate_map.end()) == 1)
            ss << "] ):";
          else 
            ss << ",\n";

          s = outfile->Append(ss.str());
          if (!s.ok()) {
            LOG(INFO) << "not able to append to: " << path << ", aborting output.";
            return;
          }
        }

        s = outfile->Close();
        if (!s.ok()) {
          LOG(INFO) << "not able to close file: " << path << ", aborting output.";
          return;
        }

      }
      LOG(INFO) << "Total candidates: " << total_candidates;
      
    }

    void Compute(OpKernelContext* ctx) override {
    
      // [idx1, idx2, min1, max1, min2, max2], [score, distance, variance], [genome1, genome2]

      const Tensor& genomes_t = ctx->input(0);
      const Tensor& match_ints_t = ctx->input(1);
      const Tensor& match_doubles_t = ctx->input(2);

      auto size = genomes_t.dim_size(0);

      //LOG(INFO) << "aggregate: processing " << size << " matches";

      auto genomes = genomes_t.matrix<string>();
      auto match_ints = match_ints_t.matrix<int32>();
      auto match_doubles = match_doubles_t.matrix<double>();

      Match match;

      for (size_t i = 0; i < size; i++) {
        if (genomes(i, 0) == "") {
          //LOG(INFO) << "empty genome at index " << i;
          break; // 
        }

        auto genome_pair = make_pair(genomes(i, 0), genomes(i, 1));
        auto seq_pair = make_pair(match_ints(i, 0), match_ints(i, 1));
        //LOG(INFO) << "idx: " << i << ", genome pair is " << genome_pair.first << ", " << genome_pair.second;
        //LOG(INFO) << "seq pair is " << seq_pair.first << ", " << seq_pair.second;
        
        match.index_1 = seq_pair.first;
        match.index_2 = seq_pair.second;
        match.seq1_min = match_ints(i, 2);
        match.seq1_max = match_ints(i, 3);
        match.seq2_min = match_ints(i, 4);
        match.seq2_max = match_ints(i, 5);
        match.score = match_doubles(i, 0);
        match.distance = match_doubles(i, 1);
        match.variance = match_doubles(i, 2);

        if (all_matches_map_.find(genome_pair) != all_matches_map_.end()) {
          auto& seq_map = all_matches_map_[genome_pair];
          auto m = seq_map.find(seq_pair);
          if (m != seq_map.end()) {

            LOG(INFO) << "found duplicate match for pairs (" << genome_pair.first << ", " 
              << genome_pair.second << ") -> (" << seq_pair.first << ", " << seq_pair.second << ")";

            if (m->second != match) {
              LOG(INFO) << "MATCHES DO NOT MATCH!! " << m->second.ToString() << "\n" << match.ToString();
            }
          }
        }
        
        all_matches_map_[genome_pair][seq_pair] = match;
      }
      
      Tensor* out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      out->scalar<string>()() = "cluster agg done";

    }

  private:

    // maintain a list of matches for each genome pair

    struct Match {
      int index_1;
      int index_2;
      int seq1_min;
      int seq1_max;
      int seq2_min;
      int seq2_max;
      double score;
      double distance;
      double variance;
      inline bool operator==(const Match& rhs) {
        return index_1 == rhs.index_1 && index_2 == rhs.index_2 && seq1_min == rhs.seq1_min &&
          seq1_max == rhs.seq1_max && seq2_min == rhs.seq2_min && seq2_max == rhs.seq2_max &&
          score == rhs.score && distance == rhs.distance && variance == rhs.variance;
      }
      inline bool operator!=(const Match& rhs) {
        return !(*this == rhs);
      }

      string ToString() {
        ostringstream s;
        s << "i1: " << index_1 << ", i2: " << index_2 << ", s1m: " << seq1_min << ", s1M: " 
          << seq1_max << ", s2m: " << seq2_min << ", s2M: " << seq2_max << ", score: "
          << score << ", dist: " << distance << ", var: " << variance;
        return s.str();
      }
    };

    // aggregate genome pairs, and sequence pairs -- because there can be dups
    unordered_map<GenomePair, unordered_map<SequencePair, Match, PairHash>, PairHash> all_matches_map_;

    Env* env_;
    string output_dir_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDClusterAggregate").Device(DEVICE_CPU), AGDClusterAggregateOp);
} //  namespace tensorflow {
