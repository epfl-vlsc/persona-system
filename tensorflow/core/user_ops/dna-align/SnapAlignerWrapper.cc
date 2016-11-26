#include <vector>
#include <exception>

#include "tensorflow/core/platform/logging.h"
#include "AlignerOptions.h"
#include "AlignmentResult.h"
#include "BaseAligner.h"
#include "Genome.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "SingleAligner.h"
#include "SeedSequencer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "SAM.h"
#include "SnapAlignerWrapper.h"

namespace snap_wrapper {
  using namespace tensorflow;
  using namespace std;

  namespace {
    const int maxReadSize = MAX_READ_LENGTH;
  }

  Status init() {
    InitializeSeedSequencers();
    return Status::OK();
  }

  PairedAligner::PairedAligner(const PairedAlignerOptions *options_, GenomeIndex *index) :
    options(options_), format(options_->useM), genome(index->getGenome()) {
    size_t memoryPoolSize = IntersectingPairedEndAligner::getBigAllocatorReservation(
                                index,
                                options->intersectingAlignerMaxHits,
                                maxReadSize,
                                index->getSeedLength(),
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                options->maxDist,
                                options->extraSearchDepth,
                                options->maxCandidatePoolSize,
                                options->maxSecondaryAlignmentsPerContig);

    memoryPoolSize += ChimericPairedEndAligner::getBigAllocatorReservation(
                                index,
                                maxReadSize,
                                options->maxHits,
                                index->getSeedLength(),
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                options->maxDist,
                                options->extraSearchDepth,
                                options->maxCandidatePoolSize,
                                options->maxSecondaryAlignmentsPerContig);

    unsigned maxPairedSecondaryHits, maxSingleSecondaryHits;

    if (options->maxSecondaryAlignmentAdditionalEditDistance < 0) {
      maxPairedSecondaryHits = 0;
      maxSingleSecondaryHits = 0;
    } else {
      LOG(ERROR) << "Enabling secondary results. This feature is not yet supported!";
      throw logic_error("Can't enable secondary results! Feature not yet supported!");
      maxPairedSecondaryHits = IntersectingPairedEndAligner::getMaxSecondaryResults(
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                maxReadSize,
                                options->maxHits,
                                index->getSeedLength(),
                                options->minSpacing,
                                options->maxSpacing);
      maxSingleSecondaryHits = ChimericPairedEndAligner::getMaxSingleEndSecondaryResults(
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                maxReadSize,
                                options->maxHits,
                                index->getSeedLength());
    }

    memoryPoolSize += (1 + maxPairedSecondaryHits) * sizeof(PairedAlignmentResult) + maxSingleSecondaryHits * sizeof(SingleAlignmentResult);

    allocator.reset(new BigAllocator(memoryPoolSize));

    if (!allocator) {
      LOG(ERROR) << "Unable to create new big allocator for pool size " << memoryPoolSize;
      throw logic_error("Allocation of big allocator failed");
    }

    auto *alloc = allocator.get();

    intersectingAligner = new (alloc) IntersectingPairedEndAligner(
                                index,
                                maxReadSize,
                                options->maxHits,
                                options->maxDist,
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                options->minSpacing,
                                options->maxSpacing,
                                options->intersectingAlignerMaxHits,
                                options->extraSearchDepth,
                                options->maxCandidatePoolSize,
                                options->maxSecondaryAlignmentsPerContig,
                                alloc,
                                options->noUkkonen,
                                options->noOrderedEvaluation,
                                options->noTruncation);

    if (!intersectingAligner) {
      auto err = "Unable to create intersecting aligner";
      LOG(ERROR) << err;
      throw logic_error(err);
    }

    aligner = new (alloc) ChimericPairedEndAligner(
                                index,
                                maxReadSize,
                                options->maxHits,
                                options->maxDist,
                                options->numSeedsFromCommandLine,
                                options->seedCoverage,
                                options->minWeightToCheck,
                                options->forceSpacing,
                                options->extraSearchDepth,
                                options->noUkkonen,
                                options->noOrderedEvaluation,
                                options->noTruncation,
                                intersectingAligner,
                                options->minReadLength,
                                options->maxSecondaryAlignmentsPerContig,
                                alloc);

    if (!aligner) {
      intersectingAligner->~IntersectingPairedEndAligner();
      auto err = "Unable to create chimeric aligner";
      LOG(ERROR) << err;
      throw logic_error(err);
    }

    allocator->checkCanaries();
  }

  PairedAligner::~PairedAligner() {
    // This calls the destructor without calling operator delete, allocator owns the memory.
    allocator->checkCanaries();
    aligner->~ChimericPairedEndAligner();
    intersectingAligner->~IntersectingPairedEndAligner();
    // No need to call delete on allocator. unique_ptr takes care of it
  }

  void
  PairedAligner::align(array<Read, 2> &snap_reads, PairedAlignmentResult &result) {
    int num_secondary_results, single_end_secondary_results_first_read,
                               single_end_secondary_results_second_read;
    aligner->align(&snap_reads[0], &snap_reads[1],
                   &result,
                   options->maxSecondaryAlignmentAdditionalEditDistance,
                   0, // secondary results buffer size
                   &num_secondary_results,
                   nullptr, // secondary results buffer
                   0, // single secondary buffer size
                   0, // maxSecondaryAlignmentsToReturn
                   // We don't use either of these, but we can't pass in nullptr
                   &single_end_secondary_results_first_read,
                   &single_end_secondary_results_second_read,
                   nullptr); // more stuff related to secondary results
  }

  Status
  PairedAligner::writeResult(array<Read, 2> &snap_reads, PairedAlignmentResult &result, AlignmentResultBuilder &result_column) {
    auto foo = foobar(2);
    array<int, 2> flags;
    // we don't write out the results yet in case one of them fails
    for (size_t i = 0; i < 2; ++i) {
      if (result.status[i] == NotFound) {
        result.location[i] = InvalidGenomeLocation;
      }
      auto &read = snap_reads[i];
      read.setAdditionalFrontClipping(0);
      TF_RETURN_IF_ERROR(adjustResults(&read,
                                       result.status[i],
                                       result.direction[i],
                                       result.mapq[i],
                                       result.location[i],
                                       format, options->useM,
                                       lvc,
                                       genome,
                                       cigars[i],
                                       flags[i]));
    }

    // Loop again now that all the adjustments worked correctly
    for (size_t i = 0; i < 2; ++i) {
      result_column.AppendAlignmentResult(result, i, cigars[i], flags[i]);
    }

    return Status::OK();
  }

  bool computeCigarFlags(
    // input
    Read *read,
    AlignmentResult &result,
    Direction &direction,
    int &map_quality,
    GenomeLocation genomeLocation, // can differ from original location
    const SAMFormat &format,
    bool useM,
    LandauVishkinWithCigar& lvc,
    const Genome* genome,
    //output
    string &cigarString,
    int &flags,
    int &addFrontClipping)
  {
    // needed for ComputeCigarString
    const int MAX_READ = MAX_READ_LENGTH;
    const int cigarBufSize = MAX_READ * 2;
    char cigarBuf[cigarBufSize];

//    const int cigarBufWithClippingSize = MAX_READ * 2 + 32;
//    char cigarBufWithClipping[cigarBufWithClippingSize];

    int editDistance = -1;

    // needed for createSAMLine
    char data[MAX_READ];
    char quality[MAX_READ];
    const char *contigName = "*";
    int contigIndex = -1;
    GenomeDistance positionInContig = 0;
    const char *mateContigName = "*";
    int mateContigIndex = -1;
    GenomeDistance matePositionInContig = 0;
    _int64 templateLength = 0;
    unsigned fullLength;
    const char* clippedData;
    unsigned clippedLength;
    unsigned basesClippedBefore;
    unsigned basesClippedAfter;
    size_t qnameLen = read->getIdLength();
    bool hasMate = false;
    bool firstInPair = false;
    bool alignedAsPair = false;
    Read *mate = NULL;
    AlignmentResult mateResult = NotFound;
    GenomeLocation mateLocation = 0;
    Direction mateDirection = FORWARD;
    GenomeDistance extraBasesClippedBefore;   // Clipping added if we align before the beginning of a chromosome

    flags = 0;
    addFrontClipping = 0;

    if (! format.createSAMLine(
        genome, &lvc,
        // output data
        data, quality, MAX_READ, contigName, contigIndex, flags, positionInContig,
        map_quality, mateContigName, mateContigIndex, matePositionInContig,
        templateLength, fullLength, clippedData, clippedLength, basesClippedBefore,
        basesClippedAfter,
        // input data
        qnameLen, read, result, genomeLocation, direction, false, // false: this is not a secondary alignment. We don't support that yet
        useM, hasMate, firstInPair, alignedAsPair, mate, mateResult, mateLocation, mateDirection,
        &extraBasesClippedBefore)) {
      return false;
    }

    if (genomeLocation != InvalidGenomeLocation) {
      // the computeCigarString method which should have been used here is private, but the
      // computeCigar method which it calls is public, so we'll use that one
      GenomeDistance extraBasesClippedAfter;
      int cigarBufUsed;
      unsigned frontHardClipping = read->getOriginalFrontHardClipping();
      unsigned backHardClipping = read->getOriginalBackHardClipping();

      format.computeCigar(
        COMPACT_CIGAR_STRING, genome, &lvc, cigarBuf, cigarBufSize, clippedData, clippedLength,
        basesClippedBefore, extraBasesClippedBefore, basesClippedAfter, &extraBasesClippedAfter,
        genomeLocation, useM, &editDistance, &cigarBufUsed, &addFrontClipping);

      if (addFrontClipping != 0) {
        return false;
      } else {
        // *o_editDistance -> editDistance
        if (editDistance == -2) {
          WriteErrorMessage( "WARNING: computeEditDistance returned -2; cigarBuf may be too small\n");
          // strcpy(cigarBufWithClipping, "*"); // already set by default
        } else if (editDistance == -1) {
          static bool warningPrinted = false;
          if (!warningPrinted) {
              WriteErrorMessage( "WARNING: computeEditDistance returned -1; this shouldn't happen\n");
              warningPrinted = true;
          }
          // strcpy(cigarBufWithClipping, "*"); // already set by default
        } else {
          // Add some CIGAR instructions for soft-clipping if we've ignored some bases in the read.
          char clipBefore[16] = {'\0'};
          char clipAfter[16] = {'\0'};
          char hardClipBefore[16] = {'\0'};
          char hardClipAfter[16] = {'\0'};

          cigarString = "";

          if (frontHardClipping > 0) {
            snprintf(hardClipBefore, sizeof(hardClipBefore), "%uH", frontHardClipping);
            cigarString += hardClipBefore;
          }
          if (basesClippedBefore + extraBasesClippedBefore > 0) {
            snprintf(clipBefore, sizeof(clipBefore), "%luS", basesClippedBefore + extraBasesClippedBefore);
            cigarString += clipBefore;
          }

          cigarString += cigarBuf;

          if (basesClippedAfter + extraBasesClippedAfter > 0) {
            snprintf(clipAfter, sizeof(clipAfter), "%luS", basesClippedAfter + extraBasesClippedAfter);
            cigarString += clipAfter;
          }
          if (backHardClipping > 0) {
            snprintf(hardClipAfter, sizeof(hardClipAfter), "%uH", backHardClipping);
            cigarString += hardClipAfter;
          }
//          snprintf(cigarBufWithClipping, cigarBufWithClippingSize, "%s%s%s%s%s", hardClipBefore, clipBefore, cigarBuf, clipAfter, hardClipAfter);
//          cigarString = cigarBufWithClipping;

        }

      }
    }

    return true;
  } // computeCigarFlags

  Status adjustResults(
                       // inputs
                       Read *read,
                       AlignmentResult &status,
                       Direction &direction,
                       int &map_quality,
                       GenomeLocation &location,
                       const SAMFormat &format,
                       bool useM,
                       LandauVishkinWithCigar& lvc,
                       const Genome* genome,
                       //outputs
                       std::string &cigarString,
                       int &flags
  ) {
    cigarString = "*"; // default value

    if (status == NotFound) {
      location = InvalidGenomeLocation;
    }

    int addFrontClipping = 0, cumulativeAddFrontClipping = 0;
    read->setAdditionalFrontClipping(0);
    GenomeLocation finalLocation = location;

    unsigned nAdjustments = 0;

    while (!computeCigarFlags(read, status, direction, map_quality, finalLocation,
                              format, useM, lvc, genome, cigarString, flags, addFrontClipping)) {
      // redo if read modified (e.g. to add soft clipping, or move alignment for a leading I.
      const Genome::Contig *originalContig = status == NotFound ? NULL
        : genome->getContigAtLocation(location);
      const Genome::Contig *newContig = status == NotFound ? NULL
        : genome->getContigAtLocation(location + addFrontClipping);
      if (newContig == NULL || newContig != originalContig || finalLocation + addFrontClipping >
        originalContig->beginningLocation + originalContig->length - genome->getChromosomePadding() ||
        nAdjustments > read->getDataLength()) {

        // Altering this would push us over a contig boundary, or we're stuck in a loop.  Just give up on the read.
        status = NotFound;
        location = InvalidGenomeLocation;
        finalLocation = InvalidGenomeLocation;
      } else {
        cumulativeAddFrontClipping += addFrontClipping;
        if (addFrontClipping > 0) {
          read->setAdditionalFrontClipping(cumulativeAddFrontClipping);
        }
        finalLocation = location + cumulativeAddFrontClipping;
      }
    } // while formatting doesn't work

    location = finalLocation;

    return Status::OK();
  } // adjustResults

  int foobar(int a) {
    return a+2;
  }

}
