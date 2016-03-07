#include <vector>

#include "snap/SNAPLib/AlignerOptions.h"
#include "snap/SNAPLib/AlignmentResult.h"
#include "snap/SNAPLib/BaseAligner.h"
#include "snap/SNAPLib/Genome.h"
#include "snap/SNAPLib/GenomeIndex.h"
#include "snap/SNAPLib/Read.h"
#include "snap/SNAPLib/SingleAligner.h"

#include "SnapAlignerWrapper.h"

namespace snap_wrapper {
    GenomeIndex* loadIndex(const char* path) {
        // 2nd and 3rd arguments are weird SNAP things that can safely be ignored
        return GenomeIndex::loadFromDirectory(const_cast<char*>(path), false, false);
    }

    BaseAligner* createAligner(GenomeIndex* index, AlignmentOptions* options) {
        return new BaseAligner(
            index,
            options->maxHitsPerSeed,
            options->maxEditDistance,
            options->maxReadLength,
            options->seedsPerRead,
            options->seedsCoverage,
            options->minimumSeedMatchesPerLocation,
            options->extraSearchDepth,
            false, false, false, // stuff that would decrease performance without impacting quality
            options->maxSecondaryAlignmentsPerContig,
            nullptr, nullptr, // Uncached Landau-Vishkin
            nullptr, // No need for stats
            nullptr // No special allocator
            );
    }

    std::vector<SingleAlignmentResult*> alignSingle(BaseAligner* aligner, AlignmentOptions* options, Read* read) {
        std::vector<SingleAlignmentResult*> results;

        if (!options->passesReadFilter(read)) {
            SingleAlignmentResult* result = new SingleAlignmentResult();
            result->status = AlignmentResult::NotFound;
            result->location = InvalidGenomeLocation;
            result->mapq = 0;
            result->direction = 0;
            results.push_back(result);
            return results;
        }

        SingleAlignmentResult* primaryResult;
        SingleAlignmentResult* secondaryResults = new SingleAlignmentResult[options->maxSecondaryAlignmentsPerRead];
        int secondaryResultsCount;

        aligner->AlignRead(
            read,
            primaryResult,
            options->maxSecondaryAlignmentEditDistance,
            options->maxSecondaryAlignmentsPerRead,
            &secondaryResultsCount,
            options->maxSecondaryAlignmentsPerRead,
            secondaryResults
            );

        if (options->passesAlignmentFilter(primaryResult->status, true)) {
            results.push_back(primaryResult);
        }

        for (int i = 0; i < secondaryResultsCount; ++i) {
            if (options->passesAlignmentFilter(secondaryResults[i].status, false)) {
                results.push_back(&secondaryResults[i]);
            }
        }

        return results;
    }
}