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
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/ChimericPairedEndAligner.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/IntersectingPairedEndAligner.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/PairedAligner.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"

namespace snap_wrapper {
    using namespace tensorflow;
    Status init();

    class PairedAligner
    {
    public:
      PairedAligner(const PairedAlignerOptions *options, GenomeIndex *index_resource);
      ~PairedAligner();

      // void for speed, as this is called per-read!
      void align(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result);
      Status writeResult(std::array<Read, 2> &snap_reads, PairedAlignmentResult &result,
                         AlignmentResultBuilder &result_column);

    private:
      std::unique_ptr<BigAllocator> allocator;
      IntersectingPairedEndAligner *intersectingAligner;
      ChimericPairedEndAligner *aligner;
      const PairedAlignerOptions *options;
      const Genome *genome;

      // members for writing out the cigar string
      SAMFormat format;
      LandauVishkinWithCigar lvc;
      // keep these around as members to avoid reallocating objects for them
      std::array<std::string, 2> cigars;
    };

    // Uses SNAP code to compute the 'cigar' and 'flags' fields from the SAM format
    bool computeCigarFlags(
                           // input
                           Read *read,
                           AlignmentResult &result,
                           Direction &direction,
                           int &map_quality,
                           GenomeLocation genomeLocation, // can differ from original location
                           const SAMFormat &format,
                           bool useM,
                           LandauVishkinWithCigar& lvc,
                           const Genome* genome,
                           //output
                           std::string &cigarString,
                           int &flags,
                           int &addFrontClipping);

    // Computes the CIGAR string and flags and adjusts location according to clipping

    tensorflow::Status adjustResults(
                                     // inputs
                                     Read *read,
                                     AlignmentResult &status,
                                     Direction &direction,
                                     int &map_quality,
                                     GenomeLocation &location,
                                     const SAMFormat &format,
                                     bool useM,
                                     LandauVishkinWithCigar& lvc,
                                     const Genome* genome,
                                     //outputs
                                     std::string &cigarString,
                                     int &flags
                                     );

    int foobar(int a );
}
