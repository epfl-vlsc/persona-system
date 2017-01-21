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
     
         two_bit_seq_ = new char[max_read_len_]; 
      }

      ~BWAAligner() {
        delete [] two_bit_seq_;
      }

      // align a whole subchunk since BWA infers insert distance from the data
      Status AlignSubchunk(ReadResource *subchunk, AlignmentResultBuilder &result_builder);

    private:
      // we dont own these
      const mem_opt_t *options_;
      const bwaidx_t *index_;
      size_t max_read_len_;
      char * two_bit_seq_;
      
      vector<mem_alnreg_v> regs_; 

      void ProcessResult(mem_aln_t* bwaresult, mem_aln_t* bwamate, format::AlignmentResult& result, string& cigar);
      Status GenerateAndInfer(ReadResource* subchunk, mem_pestat_t pes[4]);

      const string placeholder = "i'm mr meeseeks look at me!";

    };
  
}
