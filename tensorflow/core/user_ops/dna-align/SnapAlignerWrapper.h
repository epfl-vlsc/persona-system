#pragma once

#include "tensorflow/core/framework/types.h"
#include <vector>
#include "AlignmentResult.h"
#include "AlignerOptions.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "FileFormat.h"

namespace snap_wrapper {
    using namespace tensorflow;
    Status init();

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options);

    Status alignSingle(BaseAligner* aligner, 
        AlignerOptions* options, Read* read, 
        std::vector<SingleAlignmentResult>* results, 
        int num_secondary_alignments, bool& first_is_primary);

    // uses slightly modified SNAP code to write results in 
    // `format` format to the provided buffer
    Status writeRead(const ReaderContext& context, 
        Read *read, SingleAlignmentResult *results, int nResults,
        bool firstIsPrimary, char* buffer, uint64 buffer_size,
        uint64* buffer_used, const FileFormat* format, 
        LandauVishkinWithCigar& lvc, const Genome* genome);
}
