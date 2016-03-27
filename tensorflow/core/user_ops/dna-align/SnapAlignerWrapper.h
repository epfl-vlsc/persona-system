#pragma once

#include <vector>
#include "AlignmentResult.h"
#include "BaseAligner.h"
#include "GenomeIndex.h"
#include "Read.h"

namespace snap_wrapper {
    enum AlignmentFilter {
        Unaligned = 0x1,
        Single = 0x2,
        Multiple = 0x4
    };

    enum FileType {UnknownFileType, SAMFile, FASTQFile, BAMFile, InterleavedFASTQFile, CRAMFile};  // Add more as needed

    struct SNAPFile {
        SNAPFile() : fileName(NULL), secondFileName(NULL), fileType(UnknownFileType), isStdio(false), omitSQLines(false) {}
        const char          *fileName;
        const char          *secondFileName;
        FileType             fileType;
        bool                 isCompressed;
        bool                 isStdio;           // Only applies to the first file for two-file inputs
        bool				 omitSQLines;		// Special undocumented option for Charles Chiu's group.  Mostly a bad idea.

        PairedReadSupplierGenerator *createPairedReadSupplierGenerator(int numThreads, bool quicklyDropUnpairedReads, const ReaderContext& context);
        ReadSupplierGenerator *createReadSupplierGenerator(int numThreads, const ReaderContext& context);
        static bool generateFromCommandLine(const char **args, int nArgs, int *argsConsumed, SNAPFile *snapFile, bool paired, bool isInput);
    };

    struct AlignmentOptions {
        // Default values pulled from SNAP options

        unsigned int maxHitsPerSeed = 300;
        unsigned int maxEditDistance = 14;

        unsigned int minReadLength = 50;
        unsigned int maxReadLength = 400;
        ReadClippingType    clipping;
        const char         *defaultReadGroup; // if not specified in input
        float               expansionFactor;
        bool                ignoreSecondaryAlignments = true; // on input, default true
        SNAPFile            outputFile;
        const char         *rgLineContents = "@RG\tID:FASTQ\tPL:Illumina\tPU:pu\tLB:lb\tSM:sm";

        // These two are mutually exclusive: either a fixed number of seeds, or a percentage of the read size
        unsigned int seedsPerRead = 25;
        float seedsCoverage = 0;

        unsigned int minimumSeedMatchesPerLocation = 1;

        // "edit distance beyond the best hit that SNAP uses to compute MAPQ"
        unsigned int extraSearchDepth = 2;

        int maxSecondaryAlignmentEditDistance;
        int maxSecondaryAlignmentsPerContig = -1; // -1 means no limit
        int maxSecondaryAlignmentsPerRead;

        AlignmentFilter alignmentFilter;


        bool passesReadFilter(Read* read) {
            return read->getDataLength() >= minReadLength && read->countOfNs() <= maxEditDistance;
        }

        bool passesAlignmentFilter(AlignmentResult result, bool isPrimary) {
            // Don't filter out secondary alignments for low MAPQ
            if (result == AlignmentResult::MultipleHits && !isPrimary && (alignmentFilter & AlignmentFilter::Single) != 0) {
                return true;
            }

            switch (result) {
            case AlignmentResult::NotFound:
            case AlignmentResult::UnknownAlignment:
                return (alignmentFilter & AlignmentFilter::Unaligned) != 0;

            case AlignmentResult::SingleHit:
                return (alignmentFilter & AlignmentFilter::Single) != 0;

            case AlignmentResult::MultipleHits:
                return (alignmentFilter & AlignmentFilter::Multiple) != 0;

            default:
                return false; // shouldn't happen!
            }
        }
    };

    GenomeIndex* loadIndex(const char* path);

    BaseAligner* createAligner(GenomeIndex* index, AlignmentOptions* options);

    tensorflow::Status alignSingle(BaseAligner* aligner, AlignmentOptions* options, Read* read, std::vector<SingleAlignmentResult>* results);
}
