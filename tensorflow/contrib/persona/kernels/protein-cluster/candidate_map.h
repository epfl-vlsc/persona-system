#pragma once

#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <string>
#include <utility>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
  
  struct PairHash {
    template <class T1, class T2>
      size_t operator()(const std::pair<T1, T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T1>{}(p.second);
        boost::hash_combine(h1, h2);
        return h1;
      }
  };

  typedef std::pair<std::string, std::string> GenomePair; // could be replaced by ints that map to genome strings
  typedef std::pair<int, int> SequencePair;

  class CandidateMap : public ResourceBase {
  
    typedef std::unordered_map<GenomePair, 
            std::unordered_map<SequencePair, bool, PairHash>, 
            PairHash> GenomeSequenceMap;

    public:

      // returns true, or false and inserts key
      bool ExistsOrInsert(const GenomePair& g, const SequencePair& s) {
        mutex_lock l(mu_);

        auto genome_pair_it = map_.find(g);
        if (genome_pair_it != map_.end()) {
          auto seq_pair_it = genome_pair_it->second.find(s);
          if (seq_pair_it != genome_pair_it->second.end()) {
            return true;
          }
        }

        map_[g][s] = true;
        return false;
      }
  
      string DebugString() override {
        return string("A CandidateMap");
      }
    
    private:
      GenomeSequenceMap map_;
      mutex mu_;
  };


}
