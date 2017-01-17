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
    array<format::AlignmentResult, 2> results;
    array<string, 2> cigars;
    // always write pair 1 before pair 2
    // make sure 'first in pair'/'second in pair' matches original data
    snap_reads[0].setAdditionalFrontClipping(0);
    snap_reads[1].setAdditionalFrontClipping(0);
    // we don't write out the results yet in case one of them fails
    int addFrontClipping;
    int cumulativePositiveAddFrontClipping[2] = { 0, 0 };
    GenomeLocation finalLocations[2];
    finalLocations[0] = result.status[0] != NotFound ? result.location[0] : InvalidGenomeLocation;
    finalLocations[1] = result.status[1] != NotFound ? result.location[1] : InvalidGenomeLocation;
    for (int i = 0; i < 2; ++i) {
      
      auto &read = snap_reads[i];

      TF_RETURN_IF_ERROR(PostProcess(genome,
                                       &lvc,
                                       &read,
                                       result.status[i],
                                       result.mapq[i],
                                       finalLocations[i],
                                       result.direction[i],
                                       false, 
                                       results[i],
                                       cigars[i],
                                       &addFrontClipping,
                                       true, // useM
                                       true,
                                       (i == 0) ? true : false,
                                       &snap_reads[1 - i],
                                       result.status[1 - i],
                                       finalLocations[1 - i],
                                       result.direction[1 - i],
                                       result.alignedAsPair));

      if (addFrontClipping != 0) {
        const Genome::Contig *originalContig = genome->getContigAtLocation(finalLocations[i]);
        const Genome::Contig *newContig = genome->getContigAtLocation(finalLocations[i] + addFrontClipping);
        if (newContig != originalContig || NULL == newContig || finalLocations[i] + addFrontClipping > originalContig->beginningLocation + originalContig->length - genome->getChromosomePadding()) {
          //
          // Altering this would push us over a contig boundary.  Just give up on the read.
          //
          result.status[i] = NotFound;
          result.location[i] = InvalidGenomeLocation;
          finalLocations[i] = InvalidGenomeLocation;
        } else {
          if (addFrontClipping > 0) {
            cumulativePositiveAddFrontClipping[i] += addFrontClipping;
            read.setAdditionalFrontClipping(cumulativePositiveAddFrontClipping[i]);
          }
          finalLocations[i] += addFrontClipping;
        }
        if (i == 1) // if i is 1 we need to redo the first one because the second has a location change
          i = -1;
      }
      //LOG(INFO) << "cigar is: " << cigars[i];
    }

    // Loop again now that all the adjustments worked correctly
    for (size_t i = 0; i < 2; ++i) {
      result_column.AppendAlignmentResult(results[i], cigars[i]);
    }

    return Status::OK();
  }


  Status WriteSingleResult(Read &snap_read, SingleAlignmentResult &result, AlignmentResultBuilder &result_column, 
      const Genome* genome, LandauVishkinWithCigar* lvc) {
    string cigar;
    format::AlignmentResult format_result;
    snap_read.setAdditionalFrontClipping(0);

    int addFrontClipping = -1;
    GenomeLocation finalLocation = result.status != NotFound ? result.location : InvalidGenomeLocation;
    unsigned nAdjustments = 0;
    int cumulativeAddFrontClipping = 0;
    while (addFrontClipping != 0) {
      addFrontClipping = 0;
      TF_RETURN_IF_ERROR(PostProcess(genome,
                                       lvc,
                                       &snap_read,
                                       result.status,
                                       result.mapq,
                                       finalLocation,
                                       result.direction,
                                       false, 
                                       format_result,
                                       cigar,
                                       &addFrontClipping,
                                       true));
      // redo if read modified (e.g. to add soft clipping, or move alignment for a leading I.
      if (addFrontClipping != 0) {
                
        nAdjustments++;
        const Genome::Contig *originalContig = result.status == NotFound ? NULL
          : genome->getContigAtLocation(result.location);
        const Genome::Contig *newContig = result.status == NotFound ? NULL
          : genome->getContigAtLocation(result.location + addFrontClipping);
        if (newContig == NULL || newContig != originalContig || finalLocation + addFrontClipping > originalContig->beginningLocation + originalContig->length - genome->getChromosomePadding() ||
            nAdjustments > snap_read.getDataLength()) {
          //
          // Altering this would push us over a contig boundary, or we're stuck in a loop.  Just give up on the read.
          //
          result.status = NotFound;
          result.location = InvalidGenomeLocation;
          finalLocation = InvalidGenomeLocation;
        } else {
          cumulativeAddFrontClipping += addFrontClipping;
          if (addFrontClipping > 0) {
            snap_read.setAdditionalFrontClipping(cumulativeAddFrontClipping);
          }
          finalLocation = result.location + cumulativeAddFrontClipping;
        }
      }
    }
    result_column.AppendAlignmentResult(format_result, cigar);
  }

  Status PostProcess(
    const Genome* genome,
    LandauVishkinWithCigar * lv,
    Read * read,
    AlignmentResult result, 
    int mapQuality,
    GenomeLocation genomeLocation,
    Direction direction,
    bool secondaryAlignment,
    format::AlignmentResult &finalResult,
    string &cigar,
    int * addFrontClipping,
    bool useM,
    bool hasMate,
    bool firstInPair,
    Read * mate, 
    AlignmentResult mateResult,
    GenomeLocation mateLocation,
    Direction mateDirection,
    bool alignedAsPair
    ) 
  {

    cigar = "*";
    const int MAX_READ = MAX_READ_LENGTH;
    char data[MAX_READ];
    char quality[MAX_READ];
    /*const int cigarBufSize = MAX_READ * 2;
    char cigarBuf[cigarBufSize];

    const int cigarBufWithClippingSize = MAX_READ * 2 + 32;
    char cigarBufWithClipping[cigarBufWithClippingSize];

    int flags = 0;
    const char *cigar = "*";
    const char *matecontigName = "*";
    int mateContigIndex = -1;
    GenomeDistance matePositionInContig = 0;
    _int64 templateLength = 0;

    char data[MAX_READ];
    char quality[MAX_READ];

    const char* clippedData;
    unsigned fullLength;
    unsigned clippedLength;
    unsigned basesClippedBefore;
    GenomeDistance extraBasesClippedBefore;   // Clipping added if we align before the beginning of a chromosome
    unsigned basesClippedAfter;
    int editDistance = -1;*/

    *addFrontClipping = 0;
    const char *contigName = "*";
    const char *matecontigName = "*";
    int contigIndex = -1;
    GenomeDistance positionInContig = 0;
    int mateContigIndex = -1;
    GenomeDistance matePositionInContig = 0;
    
    GenomeDistance extraBasesClippedBefore;   // Clipping added if we align before the beginning of a chromosome
    _int64 templateLength = 0;
    const char* clippedData;
    unsigned fullLength;
    unsigned clippedLength;
    unsigned basesClippedBefore;
    unsigned basesClippedAfter;
    int editDistance = -1;
    uint16_t flags = 0;
    GenomeLocation orig_location = genomeLocation;

    if (secondaryAlignment) {
        flags |= SAM_SECONDARY;
    }

    //
    // If the aligner said it didn't find anything, treat it as such.  Sometimes it will emit the
    // best match that it found, even if it's not within the maximum edit distance limit (but will
    // then say NotFound).  Here, we force that to be SAM_UNMAPPED.
    //
    if (NotFound == result) {
        genomeLocation = InvalidGenomeLocation;
    }

    if (InvalidGenomeLocation == genomeLocation) {
        //
        // If it's unmapped, then always emit it in the forward direction.  This is necessary because we don't even include
        // the SAM_REVERSE_COMPLEMENT flag for unmapped reads, so there's no way to tell that we reversed it.
        //
        direction = FORWARD;
    }
    
    clippedLength = read->getDataLength();
    fullLength = read->getUnclippedLength();

    if (direction == RC) {
      for (unsigned i = 0; i < fullLength; i++) {
        data[fullLength - 1 - i] = COMPLEMENT[read->getUnclippedData()[i]];
        quality[fullLength - 1 - i] = read->getUnclippedQuality()[i];
      }
      clippedData = &data[fullLength - clippedLength - read->getFrontClippedLength()];
      basesClippedBefore = fullLength - clippedLength - read->getFrontClippedLength();
      basesClippedAfter = read->getFrontClippedLength();
    } else {
      clippedData = read->getData();
      basesClippedBefore = read->getFrontClippedLength();
      basesClippedAfter = fullLength - clippedLength - basesClippedBefore;
    }

    if (genomeLocation != InvalidGenomeLocation) {
      if (direction == RC) {
        flags |= SAM_REVERSE_COMPLEMENT;
      }
      const Genome::Contig *contig = genome->getContigForRead(genomeLocation, read->getDataLength(), &extraBasesClippedBefore);
      _ASSERT(NULL != contig && contig->length > genome->getChromosomePadding());
      genomeLocation += extraBasesClippedBefore;

      contigName = contig->name;
      contigIndex = (int)(contig - genome->getContigs());
      positionInContig = genomeLocation - contig->beginningLocation + 1; // SAM is 1-based
      mapQuality = max(0, min(70, mapQuality));       // FIXME: manifest constant.
    } else {
      flags |= SAM_UNMAPPED;
      mapQuality = 0;
      extraBasesClippedBefore = 0;
    }


    if (hasMate) {
      flags |= SAM_MULTI_SEGMENT;
      flags |= (firstInPair ? SAM_FIRST_SEGMENT : SAM_LAST_SEGMENT);
      if (mateLocation != InvalidGenomeLocation) {
        GenomeDistance mateExtraBasesClippedBefore;
        const Genome::Contig *mateContig = genome->getContigForRead(mateLocation, mate->getDataLength(), &mateExtraBasesClippedBefore);
        mateLocation += mateExtraBasesClippedBefore;
        matecontigName = mateContig->name;
        mateContigIndex = (int)(mateContig - genome->getContigs());
        matePositionInContig = mateLocation - mateContig->beginningLocation + 1;

        if (mateDirection == RC) {
          flags |= SAM_NEXT_REVERSED;
        }

        if (genomeLocation == InvalidGenomeLocation) {
          //
          // The SAM spec says that for paired reads where exactly one end is unmapped that the unmapped
          // half should just have RNAME and POS copied from the mate.
          //
          contigName = matecontigName;
          contigIndex = mateContigIndex;
          matecontigName = "=";
          positionInContig = matePositionInContig;
        }

      } else {
        flags |= SAM_NEXT_UNMAPPED;
        //
        // The mate's unmapped, so point it at us.
        //  in AGD this doesnt matter
        matecontigName = "=";
        mateContigIndex = contigIndex;
        matePositionInContig = positionInContig;
      }

      if (genomeLocation != InvalidGenomeLocation && mateLocation != InvalidGenomeLocation) {
        if (alignedAsPair) {
          flags |= SAM_ALL_ALIGNED;
        }
        // Also compute the length of the whole paired-end string whose ends we saw. This is slightly
        // tricky because (a) we may have clipped some bases before/after each end and (b) we need to
        // give a signed result based on whether our read is first or second in the pair.
        GenomeLocation myStart = genomeLocation - basesClippedBefore;
        GenomeLocation myEnd = genomeLocation + clippedLength + basesClippedAfter;
        _int64 mateBasesClippedBefore = mate->getFrontClippedLength();
        _int64 mateBasesClippedAfter = mate->getUnclippedLength() - mate->getDataLength() - mateBasesClippedBefore;
        GenomeLocation mateStart = mateLocation - (mateDirection == RC ? mateBasesClippedAfter : mateBasesClippedBefore);
        GenomeLocation mateEnd = mateLocation + mate->getDataLength() + (mateDirection == FORWARD ? mateBasesClippedAfter : mateBasesClippedBefore);
        if (contigName == matecontigName) { // pointer (not value) comparison, but that's OK.
          if (myStart < mateStart) {
            templateLength = mateEnd - myStart;
          } else {
            templateLength = -(myEnd - mateStart);
          }
        } // otherwise leave TLEN as zero.
      }

      if (contigName == matecontigName) {
        matecontigName = "=";     // SAM Spec says to do this when they're equal (and not *, which won't happen because this is a pointer, not string, compare)
      }
    }

    finalResult.location_ = genomeLocation;
    finalResult.mapq_ = mapQuality;
    finalResult.flag_ = flags;
    finalResult.next_location_ = mateLocation;
    finalResult.template_length_ = templateLength;

    const int cigarBufSize = MAX_READ * 2;
    char cigarBuf[cigarBufSize];

    const int cigarBufWithClippingSize = MAX_READ * 2 + 32;
    char cigarBufWithClipping[cigarBufWithClippingSize];

    if (orig_location != InvalidGenomeLocation) {
      const char * thecigar = SAMFormat::computeCigarString(genome, lv, cigarBuf, cigarBufSize, cigarBufWithClipping, cigarBufWithClippingSize,
        clippedData, clippedLength, basesClippedBefore, extraBasesClippedBefore, basesClippedAfter, 
        read->getOriginalFrontHardClipping(), read->getOriginalBackHardClipping(), orig_location, direction, useM,
        &editDistance, addFrontClipping);

      LOG(INFO) << "cigar output was : " << thecigar << " and frontclipping was " << *addFrontClipping;

      if (*addFrontClipping != 0) {
        // higher up the call stack deals with this
        //return errors::Internal("something went horribly wrong creating a cigar string");
      } else
        cigar = thecigar;
    }

    return Status::OK();
  }

}
