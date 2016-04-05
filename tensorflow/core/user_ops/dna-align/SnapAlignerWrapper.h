#pragma once

#include <vector>
#include "AlignmentResult.h"
#include "AlignerOptions.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"

namespace snap_wrapper {
    tensorflow::Status init();

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options);

    tensorflow::Status alignSingle(BaseAligner* aligner, AlignerOptions* options, Read* read, 
        std::vector<SingleAlignmentResult>* results, int num_secondary_alignments, bool& first_is_primary);
}
