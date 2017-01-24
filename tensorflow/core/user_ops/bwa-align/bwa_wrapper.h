#pragma once

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/agd-format/read_resource.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "bwa/bwamem.h"
#include "bwa/bwa.h"
#include "bwa/bwt.h"

namespace bwa_wrapper {
    using namespace tensorflow;
    using namespace std;

    class BWAAligner
    {
    public:
      BWAAligner(const mem_opt_t *options, const bwaidx_t *index_resource, size_t max_read_len) :
        index_(index_resource), options_(options), max_read_len_(max_read_len) {
   
          seq = new char[max_read_len];
          seqmate = new char[max_read_len];
      }

      ~BWAAligner() {
        delete [] seq;
        delete [] seqmate;
      }

      // align a whole subchunk since BWA infers insert distance from the data
      Status AlignSubchunk(ReadResource *subchunk, size_t index, vector<mem_alnreg_v>& regs);
      
      Status FinalizeSubchunk(ReadResource *subchunk, size_t regs_index, vector<mem_alnreg_v>& regs, 
          mem_pestat_t pes[4], AlignmentResultBuilder &result_builder);

    private:
      // we dont own these
      const mem_opt_t *options_;
      const bwaidx_t *index_;
      size_t max_read_len_;
      char * seq;
      char * seqmate;
      
      void ProcessResult(mem_aln_t* bwaresult, mem_aln_t* bwamate, format::AlignmentResult& result, string& cigar);

      const string placeholder = "i'm mr meeseeks look at me!";

    };
  
}
