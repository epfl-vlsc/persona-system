#include <vector>

#include "AlignerOptions.h"
#include "AlignmentResult.h"
#include "BaseAligner.h"
#include "Genome.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "SingleAligner.h"
#include "tensorflow/core/lib/core/status.h"

#include "SnapAlignerWrapper.h"

namespace snap_wrapper {
    GenomeIndex* loadIndex(const char* path) {
        // 1st argument is non-const for no reason, it's not actually modified
        // 2nd and 3rd arguments are weird SNAP things that can safely be ignored
        return GenomeIndex::loadFromDirectory(const_cast<char*>(path), false, false);
    }
    
    BaseAligner* createAligner(GenomeIndex* index, AlignerOptions* options) {
        return new BaseAligner(
            index,
            options->maxHits,
            options->maxDist,
            MAX_READ_LENGTH,
            options->numSeedsFromCommandLine,
            options->seedCoverage,
            options->minWeightToCheck,
            options->extraSearchDepth,
            false, false, false, // stuff that would decrease performance without impacting quality
            options->maxSecondaryAlignmentsPerContig,
            nullptr, nullptr, // Uncached Landau-Vishkin
            nullptr, // No need for stats
            nullptr // No special allocator
        );
    }

    bool passesReadFilter(Read* read, AlignerOptions* options) {
      return read->getDataLength() >= options->minReadLength && read->countOfNs() <= options->maxDist;
    }

    tensorflow::Status alignSingle(BaseAligner* aligner, AlignerOptions* options, Read* read, 
        std::vector<SingleAlignmentResult>* results, int num_secondary_alignments) {
        if (!passesReadFilter(read, options)) {
            SingleAlignmentResult result;
            result.status = AlignmentResult::NotFound;
            result.location = InvalidGenomeLocation;
            result.mapq = 0;
            result.direction = 0;
            results->push_back(result);
            return tensorflow::Status::OK();
        }

        SingleAlignmentResult primaryResult;
        SingleAlignmentResult* secondaryResults = new SingleAlignmentResult[num_secondary_alignments];
        int secondaryResultsCount;

        aligner->AlignRead(
            read,
            &primaryResult,
            options->maxSecondaryAlignmentAdditionalEditDistance,
            num_secondary_alignments,
            &secondaryResultsCount,
            num_secondary_alignments,
            secondaryResults
        );
        
        if (options->passFilter(read, primaryResult.status, false, false)) {
            results->push_back(primaryResult);
        }

        for (int i = 0; i < secondaryResultsCount; ++i) {
            if (options->passFilter(read, secondaryResults[i].status, false, true)) {
                results->push_back(secondaryResults[i]);
            }
        }

        delete[] secondaryResults;

        return tensorflow::Status::OK();
    }
}
