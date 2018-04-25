#pragma once

#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <string>
#include <utility>

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
  typedef std::unordered_map<GenomePair, 
          std::unordered_map<SequencePair, bool, PairHash>, 
          PairHash> GenomeSequenceMap;

}
