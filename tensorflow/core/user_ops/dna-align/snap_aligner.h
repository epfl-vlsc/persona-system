

#include "./snap/SNAPLib/AlignerStats.h"
#include "./snap/SNAPLib/AlignmentResult.h"
#include "./snap/SNAPLib/AlignerOptions.h"

struct SnapAlignerParams {

    // I don't know how many of these are actually needed
    // perhaps we can fold the required AlignerOptions fields into this
    _int64                               alignStart;
    _int64                               alignTime;
    AlignerOptions                      *options;
    AlignerStats                        *stats;
    AlignerExtension                    *extension;
    unsigned                             maxDist;
    unsigned                             numSeedsFromCommandLine;
    double                               seedCoverage;
    unsigned                             minWeightToCheck;
    int                                  maxHits;
    bool                                 detailedStats;
    unsigned                             extraSearchDepth;
    const char                          *version;
    FILE                                *perfFile;
    bool                                 noUkkonen;
    bool                                 noOrderedEvaluation;
    bool								 noTruncation;
    int                                  maxSecondaryAlignmentAdditionalEditDistance;
    int									 maxSecondaryAlignments;
    int                                  maxSecondaryAlignmentsPerContig;
    unsigned							 minReadLength;

};

class SnapAligner {
   
    public:
        // need to use opkernelcontext allocator?
        SnapAligner(SnapAlignerParams* params) : params_(params);
        ~SnapAligner() { delete params; }

        // align() performs what runIterationThread does in SNAP
        Status align(Read* read, GenomeIndex* genome_index, std::vector<SingleAlignmentResult>* alignment_results);

    private:

        SnapAlignerParams* params_; // owned
        BaseAligner* base_aligner_;

};

