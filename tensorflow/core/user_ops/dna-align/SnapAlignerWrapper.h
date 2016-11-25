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
      SAMFormat format;
      std::string cigar;
    };

    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options);

    // Uses SNAP code to compute the 'cigar' and 'flags' fields from the SAM format
    bool computeCigarFlags(
        // inputs
        Read *read,
        SingleAlignmentResult* results,
        int whichResult,
        bool firstIsPrimary, 
        GenomeLocation genomeLocation,
        const SAMFormat &format,
        bool useM,
        LandauVishkinWithCigar& lvc, 
        const Genome* genome,
        //outputs
        std::string &cigarString,
        int &flags,
        int &addFrontClipping);

    // Computes the CIGAR string and flags and adjusts location according to clipping
    tensorflow::Status adjustResults(
        // inputs
        Read *read,
        SingleAlignmentResult& result,
        bool firstIsPrimary, 
        const SAMFormat &format,
        bool useM,
        LandauVishkinWithCigar& lvc, 
        const Genome* genome,
        //outputs
        std::string &cigarString,
        int &flags);


    // uses slightly modified SNAP code to write results in 
    // `format` format to the provided buffer
    Status writeRead(const ReaderContext& context, 
        Read *read, SingleAlignmentResult *results, int nResults,
        bool firstIsPrimary, char* buffer, uint64 buffer_size,
        uint64* buffer_used, const FileFormat* format, 
        LandauVishkinWithCigar& lvc, const Genome* genome);
}
