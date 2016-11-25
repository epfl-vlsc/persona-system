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

  tensorflow::Status init() {
    InitializeSeedSequencers();
    return tensorflow::Status::OK();
  }

  PairedAligner::PairedAligner(const PairedAlignerOptions *options_, GenomeIndex *index) : options(options_), format(options_->useM) {
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
      LOG(WARNING) << "Enabling secondary results. This feature is not yet supported!";
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
    // First, we set invalid location on the pairs if the aligner couldn't find the result
    for (size_t i = 0; i < 2; ++i) {
      if (result.status[i] == NotFound) {
        result.location[i] = InvalidGenomeLocation;
      }
    }
    for (auto &read : snap_reads) {
      read.setAdditionalFrontClipping(0);
    }

    // compute the CIGAR strings for both

    return Status::OK();
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

  bool computeCigarFlags(
    // input
    Read *read,
    SingleAlignmentResult* results,
    int whichResult, // to avoid passing too many parameters
    bool firstIsPrimary,
    GenomeLocation genomeLocation, // can differ from original location
    const SAMFormat &format,
    bool useM,
    LandauVishkinWithCigar& lvc, 
    const Genome* genome,
		//output
		std::string &cigarString,
    int &flags,
		int &addFrontClipping
  ) 
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
    AlignmentResult result = results[whichResult].status;
    Direction direction = results[whichResult].direction;
    bool secondaryAlignment = (whichResult > 0) || !firstIsPrimary;

    if (! format.createSAMLine(
        genome, &lvc,
        // output data
        data, quality, MAX_READ, contigName, contigIndex, flags, positionInContig, 
        results[whichResult].mapq, mateContigName, mateContigIndex, matePositionInContig,
        templateLength, fullLength, clippedData, clippedLength, basesClippedBefore,
        basesClippedAfter,
        // input data
        qnameLen, read, result, genomeLocation, direction, secondaryAlignment,
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

	tensorflow::Status adjustResults(
		// input
    Read *read,
    SingleAlignmentResult& result,
    bool firstIsPrimary,
    const SAMFormat &format,
    bool useM,
    LandauVishkinWithCigar& lvc, 
    const Genome* genome,
		//output
		std::string &cigarString,
    int &flags
	) {
    cigarString = "*"; // default value

    if (result.status == NotFound) {
      result.location = InvalidGenomeLocation;
    }

    GenomeLocation finalLocation;

    int addFrontClipping = 0;
    read->setAdditionalFrontClipping(0);
    int cumulativeAddFrontClipping = 0;
    finalLocation = result.location;

    unsigned nAdjustments = 0;

    while (!computeCigarFlags(read, &result, 0, firstIsPrimary, finalLocation, format, useM, 
      lvc, genome, cigarString, flags, addFrontClipping)) {
      
      // redo if read modified (e.g. to add soft clipping, or move alignment for a leading I.
      const Genome::Contig *originalContig = result.status == NotFound ? NULL
        : genome->getContigAtLocation(result.location);
      const Genome::Contig *newContig = result.status == NotFound ? NULL
        : genome->getContigAtLocation(result.location + addFrontClipping);
      if (newContig == NULL || newContig != originalContig || finalLocation + addFrontClipping > 
        originalContig->beginningLocation + originalContig->length - genome->getChromosomePadding() ||
        nAdjustments > read->getDataLength()) {
        
        // Altering this would push us over a contig boundary, or we're stuck in a loop.  Just give up on the read.
        result.status = NotFound;
        result.location = InvalidGenomeLocation;
        finalLocation = InvalidGenomeLocation;
      } else {
        cumulativeAddFrontClipping += addFrontClipping;
        if (addFrontClipping > 0) {
          read->setAdditionalFrontClipping(cumulativeAddFrontClipping);
        }
        finalLocation = result.location + cumulativeAddFrontClipping;
      }
    } // while formatting doesn't work			

    result.location = finalLocation; 

    return tensorflow::Status::OK();
	} // adjustResults

  tensorflow::Status writeRead(const ReaderContext& context, 
      Read *read, SingleAlignmentResult *results, int nResults,
      bool firstIsPrimary, char* buffer, uint64 buffer_size,
      uint64* buffer_used, const FileFormat* format,
      LandauVishkinWithCigar& lvc, const Genome* genome) {

    //LOG(INFO) << "SnapWrapper writing read! buffer_size=" << 
    //  buffer_size << ", nResults=" << nResults;
    uint64 used = 0;

    for (int i = 0; i < nResults; i++) {
      if (results[i].status == NotFound) {
        results[i].location = InvalidGenomeLocation;
      }
    }

    GenomeLocation finalLocation;

    // this is adapted from SNAP, some things are a little arcane
    for (int whichResult = 0; whichResult < nResults; whichResult++) {
      int addFrontClipping = 0;
      read->setAdditionalFrontClipping(0);
      int cumulativeAddFrontClipping = 0;
      finalLocation = results[whichResult].location;

      unsigned nAdjustments = 0;
      size_t used_local;
      while (!format->writeRead(context, &lvc, buffer + used, buffer_size - used,
            &used_local, read->getIdLength(), read, 
            results[whichResult].status,results[whichResult].mapq, 
            finalLocation, results[whichResult].direction, 
            (whichResult > 0) || !firstIsPrimary, &addFrontClipping)) {

        nAdjustments++;

        if (0 == addFrontClipping) {
          // *this* is how SNAP notifies you there wasn't enough 
          // space in the buffer. Much intuitive, very logic, wow. 
          // caller will call again with fresh buffer
          return tensorflow::errors::ResourceExhausted("buffer too full in SNAP writeRead"); 
        }

        // redo if read modified (e.g. to add soft clipping, or move alignment for a leading I.
        const Genome::Contig *originalContig = results[whichResult].status == NotFound ? NULL
          : genome->getContigAtLocation(results[whichResult].location);
        const Genome::Contig *newContig = results[whichResult].status == NotFound ? NULL
          : genome->getContigAtLocation(results[whichResult].location + addFrontClipping);
        if (newContig == NULL || newContig != originalContig || finalLocation
            + addFrontClipping > originalContig->beginningLocation 
            + originalContig->length - genome->getChromosomePadding() ||
            nAdjustments > read->getDataLength()) {
          //
          // Altering this would push us over a contig boundary, or we're stuck in a loop.  Just give up on the read.
          //
          results[whichResult].status = NotFound;
          results[whichResult].location = InvalidGenomeLocation;
          finalLocation = InvalidGenomeLocation;
        } else {
          cumulativeAddFrontClipping += addFrontClipping;
          if (addFrontClipping > 0) {
            read->setAdditionalFrontClipping(cumulativeAddFrontClipping);
          }
          finalLocation = results[whichResult].location + cumulativeAddFrontClipping;
        }
      } // while formatting doesn't work
      //LOG(INFO) << "Formatting was successful, used " << used_local << " bytes";
      used += used_local;
      if (used > buffer_size) {
        // shouldn't happen, but of course is error
        return tensorflow::errors::Internal("Buffer overflow in Snapwrapper read writer?");
      }

      if (used > 0xffffffff) {
        return tensorflow::errors::Internal("Snap Read writer:writeReads: used too big");
      }
    } // for each result.

    read->setAdditionalFrontClipping(0);
    *buffer_used = used;
    return tensorflow::Status::OK();
  }
}

