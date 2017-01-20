#pragma once

#include "tensorflow/core/framework/types.h"
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/ChimericPairedEndAligner.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/IntersectingPairedEndAligner.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/BWAAligner.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"

namespace bwa_wrapper {
    using namespace tensorflow;
    Status init();

    class BWAAligner
    {
    public:
      BWAAligner(const BWAAlignerOptions *options, BWAIndex *index_resource);
      ~BWAAligner();

      // void for speed, as this is called per-read!
      void align(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result);
      Status writeResult(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result,
                         AlignmentResultBuilder &result_column);

    private:
      std::unique_ptr<BigAllocator> allocator;
      IntersectingPairedEndAligner *intersectingAligner;
      ChimericPairedEndAligner *aligner;
      const BWAAlignerOptions *options;
      const Genome *genome;

      // members for writing out the cigar string
      SAMFormat format;
      LandauVishkinWithCigar lvc;
      // keep these around as members to avoid reallocating objects for them
      std::array<std::string, 2> cigars;
    };
  
    Status WriteSingleResult(Read &snap_read, SingleAlignmentResult &result, AlignmentResultBuilder &result_column, 
      const Genome* genome, LandauVishkinWithCigar* lvc);
  
    Status PostProcess(
      const Genome* genome,
      LandauVishkinWithCigar * lv,
      Read * read,
      AlignmentResult result, 
      int mapQuality,
      GenomeLocation genomeLocation,
      Direction direction,
      bool secondaryAlignment,
      format::AlignmentResult &finalResult,
      string &cigar,
      int * addFrontClipping,
      bool useM,
      bool hasMate = false,
      bool firstInPair = false,
      Read * mate = NULL, 
      AlignmentResult mateResult = NotFound,
      GenomeLocation mateLocation = 0,
      Direction mateDirection = FORWARD,
      bool alignedAsPair = false
      );
}
