#pragma once

#include <vector>
#include "snap/SNAPLib/AlignmentResult.h"
#include "snap/SNAPLib/BaseAligner.h"
#include "snap/SNAPLib/GenomeIndex.h"
#include "snap/SNAPLib/Read.h"

namespace snap_wrapper {
    enum AlignmentFilter {
        Unaligned = 0x1,
        Single = 0x2,
        Multiple = 0x4
    };

    struct AlignmentOptions {
        int maxHitsPerSeed;
        int maxEditDistance;

        int minReadLength;
        int maxReadLength;

        // These two are mutually exclusive: either a fixed number of seeds, or a percentage of the read size
        int seedsPerRead;
        float seedsCoverage;

        int minimumSeedMatchesPerLocation;

        // "edit distance beyond the best hit that SNAP uses to compute MAPQ"
        int extraSearchDepth;

        int maxSecondaryAlignmentEditDistance;
        int maxSecondaryAlignmentsPerContig;
        int maxSecondaryAlignmentsPerRead;

        AlignmentFilter alignmentFilter;


        bool passesReadFilter(Read* read) {
            return read->getDataLength() >= minReadLength && read->countOfNs() <= maxEditDistance;
        }

        bool passesAlignmentFilter(AlignmentResult result, bool isPrimary) {
            // Don't filter out secondary alignments for low MAPQ
            if (result == MultipleHits && !isPrimary && (alignmentFilter & AlignmentFilter::Single) != 0) {
                return true;
            }

            switch (result) {
            case NotFound:
            case UnknownAlignment:
                return (alignmentFilter & AlignmentFilter::Unaligned) != 0;

            case SingleHit:
                return (alignmentFilter & AlignmentFilter::Single) != 0;

            case MultipleHits:
                return (alignmentFilter & AlignmentFilter::Multiple) != 0;

            default:
                return false; // shouldn't happen!
            }
        }
    };

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignmentOptions* options);

    std::vector<SingleAlignmentResult*> alignSingle(BaseAligner* aligner, AlignmentOptions* options, Read* read);
}