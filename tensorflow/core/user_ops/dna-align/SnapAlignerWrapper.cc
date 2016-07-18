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
#include "SAM.h"
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

  tensorflow::Status computeCigarFlags(
    // input
    Read *read,
    std::vector<SingleAlignmentResult> &results,
    int nResults,
    bool firstIsPrimary,
    const SAMFormat &format,
    bool useM,
    LandauVishkinWithCigar& lvc, 
    const Genome* genome,
		//output
		std::string &cigarString,
    int &flags
  ) 
  {  
    // Adapted from SNAP, but not using the writeRead method, as we need only
    // the cigar string and the flag, not also writing the output to the buffer
    
    for (int i = 0; i < nResults; i++) {
      if (results[i].status == NotFound) {
        results[i].location = InvalidGenomeLocation;
      }
    }

    // needed for ComputeCigarString
    const int MAX_READ = MAX_READ_LENGTH;
    const int cigarBufSize = MAX_READ * 2;
    char cigarBuf[cigarBufSize];

    const int cigarBufWithClippingSize = MAX_READ * 2 + 32;
    char cigarBufWithClipping[cigarBufWithClippingSize];

    int editDistance = -1;
    int *o_addFrontClipping = new int;
    *o_addFrontClipping = 0;

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

    for (int whichResult = 0; whichResult < nResults; whichResult++) {
      bool status;
      read->setAdditionalFrontClipping(0);
			
			AlignmentResult result = results[whichResult].status; 
      GenomeLocation genomeLocation = results[whichResult].location;
			Direction direction = results[whichResult].direction;
			bool secondaryAlignment = (whichResult > 0) || !firstIsPrimary;

      flags = 0;

	    status = format.createSAMLine(
  	    genome, &lvc,
    	  // output data
      	data, quality, MAX_READ, contigName, contigIndex, flags, positionInContig, 
				results[whichResult].mapq, mateContigName, mateContigIndex, matePositionInContig,
				templateLength, fullLength, clippedData, clippedLength, basesClippedBefore,
				basesClippedAfter,
				// input data
 			  qnameLen, read, result, genomeLocation, direction, secondaryAlignment,
				useM, hasMate, firstInPair, alignedAsPair, mate, mateResult, mateLocation, mateDirection,
        &extraBasesClippedBefore);

      if (!status) {
        return tensorflow::errors::Internal("createSAMLine failed!"); // TODO: check if right type of error
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
          genomeLocation, useM, &editDistance, &cigarBufUsed, o_addFrontClipping);
          
      	/*if (*o_addFrontClipping != 0) {
          // TODO: check type of error
          return tensorflow::errors::ResourceExhausted("buffer too full in SNAP writeRead"); 
    		}*/

				// *o_editDistance -> editDistance
				if (editDistance == -2) {
						WriteErrorMessage( "WARNING: computeEditDistance returned -2; cigarBuf may be too small\n");
						strcpy(cigarBufWithClipping, "*");
						// return "*";
				} else if (editDistance == -1) {
						static bool warningPrinted = false;
						if (!warningPrinted) {
								WriteErrorMessage( "WARNING: computeEditDistance returned -1; this shouldn't happen\n");
								warningPrinted = true;
						}
						strcpy(cigarBufWithClipping, "*");
						// return "*";
				} else {
						// Add some CIGAR instructions for soft-clipping if we've ignored some bases in the read.
						char clipBefore[16] = {'\0'};
						char clipAfter[16] = {'\0'};
						char hardClipBefore[16] = {'\0'};
						char hardClipAfter[16] = {'\0'};
						if (frontHardClipping > 0) {
								snprintf(hardClipBefore, sizeof(hardClipBefore), "%uH", frontHardClipping);
						}
						if (basesClippedBefore + extraBasesClippedBefore > 0) {
								snprintf(clipBefore, sizeof(clipBefore), "%lluS", basesClippedBefore + extraBasesClippedBefore);
						}
						if (basesClippedAfter + extraBasesClippedAfter > 0) {
								snprintf(clipAfter, sizeof(clipAfter), "%lluS", basesClippedAfter + extraBasesClippedAfter);
						}
						if (backHardClipping > 0) {
								snprintf(hardClipAfter, sizeof(hardClipAfter), "%uH", backHardClipping);
						}
						snprintf(cigarBufWithClipping, cigarBufWithClippingSize, "%s%s%s%s%s", hardClipBefore, clipBefore, cigarBuf, clipAfter, hardClipAfter);
				}
				
				// maybe TODO: validate Cigar String - cannot access validateCigarString (private member and only for debugging)

				cigarString = cigarBufWithClipping;	
      }
		}

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

