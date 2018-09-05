
#pragma once

namespace tensorflow {

struct Parameters {
  int min_score;
  bool subsequence_homology;
  int max_representatives;
  int max_n_aa_not_covered;
};

}
