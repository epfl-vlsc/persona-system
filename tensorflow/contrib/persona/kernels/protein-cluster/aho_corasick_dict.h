
#pragma once

#include <cmath>
#include <experimental/string_view>
#include <unordered_set>
#include "tensorflow/contrib/persona/kernels/protein-cluster/aho_corasick/aho_corasick.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// parametrized by Kmer size K
template <size_t K>
class AhoCorasickDict {
  using KmerSet = std::unordered_set<std::experimental::string_view>;
  using ProteinACTrie = ACTrie<26, false>;

 public:
  // add another rep to this dict
  // rep must live as long as this object lives
  void AddRepresentative(const std::string& rep) {
    BuildOrUpdateKmerSet(&kmer_set_, rep.c_str(),
                         rep.size());  // update set with rep kmers
    // rebuild the trie
    ac_trie_ = ProteinACTrie();
    ac_trie_.Initialize(kmer_set_, sequence_length_sum_);
    /*LOG(INFO) << "Building AH dict for seq of len " << rep.size() << ", had "
              << kmer_set_.size() << " unique kmers and a total length sum of "
              << sequence_length_sum_;*/
    // results.reserve(32);
  }

  // for the given query, compute the jaccard coefficient
  float Jaccard(const char* query, size_t len) {
    // KmerSet query_set;
    // BuildOrUpdateKmerSet(&query_set, query, len);

    // TODO modify interface to avoid string copy here
    // results.clear();
    // ah_trie_.parse_text(std::string(query, len), results);

    auto results_size =
        ac_trie_.Search(std::experimental::string_view(query, len));

    // auto total_kmers_query = query_set.size();  // approximate with seq len
    auto total_kmers = kmer_set_.size();

    auto r = float(total_kmers + len - results_size);
    auto denom = r >= 1000.0f ? 1000.0f : r;
    float jaccard = float(results_size) / denom;

    return -100.0 / K * log((2.0f * jaccard) / (1.0f + jaccard));
  }

 private:
  // build a set of unique kmers from a sequence
  void BuildOrUpdateKmerSet(KmerSet* set, const char* rep, size_t len) {
    auto p = rep;
    while (p < (rep + len - K + 1)) {
      set->insert(std::experimental::string_view(p, K));
      // cout << "inserting: " << string_view(p, 3) << "\n";
      p++;
    }
    sequence_length_sum_ = K * set->size();
    ;
  };

  uint32_t sequence_length_sum_ = 0;
  ProteinACTrie ac_trie_;
  KmerSet kmer_set_;
  // aho_corasick::trie::emit_collection results;
};

}  // namespace tensorflow