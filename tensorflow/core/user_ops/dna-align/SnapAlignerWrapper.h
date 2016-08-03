#pragma once

#include "tensorflow/core/framework/types.h"
#include <vector>
#include "AlignmentResult.h"
#include "AlignerOptions.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "FileFormat.h"
#include "SAM.h" 

namespace snap_wrapper {
    using namespace tensorflow;
    Status init();

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options);

    Status alignSingle(BaseAligner* aligner, 
        AlignerOptions* options, Read* read, 
        std::vector<SingleAlignmentResult>* results, 
        int num_secondary_alignments, bool& first_is_primary);
 
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
