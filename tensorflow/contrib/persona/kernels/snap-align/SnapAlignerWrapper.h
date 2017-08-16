#pragma once

#include "tensorflow/core/framework/types.h"
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "AlignmentResult.h"
#include "AlignerOptions.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "FileFormat.h"
#include "SAM.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/ChimericPairedEndAligner.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/IntersectingPairedEndAligner.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/PairedAligner.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"

namespace snap_wrapper {
    using namespace tensorflow;
    Status init();

    class PairedAligner
    {
    public:
      PairedAligner(const PairedAlignerOptions *options, GenomeIndex *index_resource);
      ~PairedAligner();

      // void for speed, as this is called per-read!
      void align(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result, int max_secondary,
          PairedAlignmentResult** secondary_results, int* num_secondary_results,
          SingleAlignmentResult** secondary_single_results, int* num_secondary_single_results_first,
          int* num_secondary_single_results_second);

      Status writeResult(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result,
                         AlignmentResultBuilder &result_column, bool is_secondary);

    private:
      std::unique_ptr<BigAllocator> allocator;
      std::unique_ptr<PairedAlignmentResult[]> secondary_results_;
      std::unique_ptr<SingleAlignmentResult[]> secondary_single_results_;
      IntersectingPairedEndAligner *intersectingAligner;
      ChimericPairedEndAligner *aligner;
      const PairedAlignerOptions *options;
      const Genome *genome;
      unsigned maxPairedSecondaryHits_, maxSingleSecondaryHits_;

      // members for writing out the cigar string
      SAMFormat format;
      LandauVishkinWithCigar lvc;
      // keep these around as members to avoid reallocating objects for them
      std::array<std::string, 2> cigars;
    };
  
    Status WriteSingleResult(Read &snap_read, SingleAlignmentResult &result, AlignmentResultBuilder &result_column, 
      const Genome* genome, LandauVishkinWithCigar* lvc, bool is_secondary, bool use_m);
  
    Status PostProcess(
      const Genome* genome,
      LandauVishkinWithCigar * lv,
      Read * read,
      AlignmentResult result, 
      int mapQuality,
      GenomeLocation genomeLocation,
      Direction direction,
      bool secondaryAlignment,
      Alignment &finalResult,
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
