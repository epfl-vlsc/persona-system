#pragma once

#include <vector>
#include "AlignmentResult.h"
#include "AlignerOptions.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"

namespace snap_wrapper {

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options);

    tensorflow::Status alignSingle(BaseAligner* aligner, AlignerOptions* options, Read* read, std::vector<SingleAlignmentResult>* results);
}
