#include <vector>

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

#include "SnapAlignerWrapper.h"

namespace snap_wrapper {
  tensorflow::Status init() {
    InitializeSeedSequencers();
    return tensorflow::Status::OK();
  }

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
      std::vector<SingleAlignmentResult>* results, int num_secondary_alignments, bool& first_is_primary) {
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
    SingleAlignmentResult* secondaryResults = nullptr;
    if (num_secondary_alignments != 0) {
      secondaryResults = new SingleAlignmentResult[num_secondary_alignments];
    }
    int secondaryResultsCount;

    aligner->AlignRead(
        read,
        &primaryResult,
        options->maxSecondaryAlignmentAdditionalEditDistance,
        num_secondary_alignments * sizeof(SingleAlignmentResult),
        &secondaryResultsCount,
        num_secondary_alignments,
        secondaryResults
        );

    /*LOG(INFO) << "Primary result: location " << primaryResult.location <<
      " direction: " << primaryResult.direction << " score " << primaryResult.score;
      LOG(INFO) << "secondaryResultsCount is " << secondaryResultsCount;*/

    if (options->passFilter(read, primaryResult.status, false, false)) {
      results->push_back(primaryResult);
      first_is_primary = true;
    }
    else {
      first_is_primary = false;
    }

    for (int i = 0; i < secondaryResultsCount; ++i) {
      if (options->passFilter(read, secondaryResults[i].status, false, true)) {
        results->push_back(secondaryResults[i]);
      }
    }

    delete[] secondaryResults;

    return tensorflow::Status::OK();
  }

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

