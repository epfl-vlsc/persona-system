//
// Created by Stuart Byma on 17/04/17.
//

#include "tensorflow/contrib/persona/kernels/snap-align/single_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  SnapSingleExecutor::SnapSingleExecutor(Env *env, GenomeIndex *index, AlignerOptions *options,
                                         int num_threads, int capacity) : index_(index),
                                                                          options_(options),
                                                                          num_threads_(num_threads),
                                                                          capacity_(capacity) {
    genome_ = index_->getGenome();
    // create a threadpool to execute stuff
    workers_.reset(new thread::ThreadPool(env, "SnapSingle", num_threads_));
    request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<ReadResource>>>(capacity));
    auto s = snap_wrapper::init();
    if (s.ok()) {
      init_workers();
    } else {
      LOG(ERROR) << "Unable to run snap_wrapper::init()";
      compute_status_ = s;
    }
  }

  SnapSingleExecutor::~SnapSingleExecutor() {
    if (!run_) {
      LOG(ERROR) << "Unable to safely wait in ~SnapAlignSingleOp for all threads. run_ was toggled to false\n";
    }
    run_ = false;
    request_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Status SnapSingleExecutor::EnqueueChunk(std::shared_ptr<ResourceContainer<ReadResource> > chunk) {
    if (!compute_status_.ok()) return compute_status_;
    if (!request_queue_->push(chunk))
      return Internal("Single executor failed to push to request queue");
    else
      return Status::OK();
  }

  void SnapSingleExecutor::init_workers() {

    auto aligner_func = [this]() {
      //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = id_.fetch_add(1, memory_order_relaxed);

      int capacity = request_queue_->capacity();

      unsigned alignmentResultBufferCount;
      if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        alignmentResultBufferCount = 1; // For the primary alignment
      } else {
        alignmentResultBufferCount =
                BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine, options_->seedCoverage,
                                                    MAX_READ_LENGTH, options_->maxHits, index_->getSeedLength()) +
                1; // +1 for the primary alignment
      }
      size_t alignmentResultBufferSize =
              sizeof(SingleAlignmentResult) * (alignmentResultBufferCount + 1); // +1 is for primary result

      unique_ptr<BigAllocator> allocator(new BigAllocator(BaseAligner::getBigAllocatorReservation(index_, true,
                                                                                                  options_->maxHits,
                                                                                                  MAX_READ_LENGTH,
                                                                                                  index_->getSeedLength(),
                                                                                                  options_->numSeedsFromCommandLine,
                                                                                                  options_->seedCoverage,
                                                                                                  options_->maxSecondaryAlignmentsPerContig)
                                                          + alignmentResultBufferSize));

      /*LOG(INFO) << "reservation: " << BaseAligner::getBigAllocatorReservation(index, true,
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig)
          + alignmentResultBufferSize;*/

      BaseAligner *base_aligner = new(allocator.get()) BaseAligner(
              index_,
              options_->maxHits,
              options_->maxDist,
              MAX_READ_LENGTH,
              options_->numSeedsFromCommandLine,
              options_->seedCoverage,
              options_->minWeightToCheck,
              options_->extraSearchDepth,
              false, false, false, // stuff that would decrease performance without impacting quality
              options_->maxSecondaryAlignmentsPerContig,
              nullptr, nullptr, // Uncached Landau-Vishkin
              nullptr, // No need for stats
              allocator.get()
      );

      allocator->checkCanaries();

      base_aligner->setExplorePopularSeeds(options_->explorePopularSeeds);
      base_aligner->setStopOnFirstHit(options_->stopOnFirstHit);

      const char *bases, *qualities;
      size_t bases_len, qualities_len;
      SingleAlignmentResult primaryResult;
      vector<SingleAlignmentResult> secondaryResults;
      secondaryResults.resize(alignmentResultBufferCount);

      int num_secondary_results;
      SAMFormat format(options_->useM);
      vector<AlignmentResultBuilder> result_builders;
      string cigarString;
      int flag;
      Read snap_read;
      LandauVishkinWithCigar lvc;
      size_t num_columns;

      vector<BufferPair *> result_bufs;
      ReadResource *subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer < ReadResource> > reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        ScopeDropIfEqual<decltype(reads_container)> scope_dropper(*request_queue_, reads_container);

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {
          num_columns = result_bufs.size();

          if (num_columns > result_builders.size()) {
            result_builders.resize(num_columns);
          }

          for (int i = 0; i < num_columns; i++) {
            result_builders[i].SetBufferPair(result_bufs[i]);
          }

          for (subchunk_status = subchunk_resource->get_next_record(snap_read); subchunk_status.ok();
               subchunk_status = subchunk_resource->get_next_record(snap_read)) {
            cigarString.clear();
            snap_read.clip(options_->clipping);
            if (snap_read.getDataLength() < options_->minReadLength || snap_read.countOfNs() > options_->maxDist) {
              primaryResult.status = AlignmentResult::NotFound;
              primaryResult.location = InvalidGenomeLocation;
              primaryResult.mapq = 0;
              primaryResult.direction = FORWARD;
              auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc,
                                                       false);

              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
              }
              for (decltype(num_columns) i = 1; i < num_columns; i++) {
                // fill the columns with empties to maintain index equivalence
                result_builders[i].AppendEmpty();
              }
              continue;
            }

            base_aligner->AlignRead(
                    &snap_read,
                    &primaryResult,
                    options_->maxSecondaryAlignmentAdditionalEditDistance,
                    alignmentResultBufferCount,
                    &num_secondary_results,
                    num_columns-1, // maximum number of secondary results
                    &secondaryResults[0] //secondaryResults
            );

            flag = 0;

            // First, write the primary results
            auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc,
                                                     false);

            if (!s.ok()) {
              LOG(ERROR) << "adjustResults did not return OK!!!";
              compute_status_ = s;
              return;
            }

            // Then write the secondary results if we specified them
            for (int i = 0; i < num_secondary_results; i++) {
              s = snap_wrapper::WriteSingleResult(snap_read, secondaryResults[i], result_builders[i + 1], genome_,
                                                  &lvc, true);
              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
                compute_status_ = s;
                return;
              }
            }
            for (int i = num_secondary_results+1; i < num_columns; i++) {
              // fill the columns with empties to maintain index equivalence
              result_builders[i].AppendEmpty();
            }
          }

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            return;
          }

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        }

        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status
                     << "\n";
          compute_status_ = io_chunk_status;
          break;
        }
      }

      base_aligner->~BaseAligner(); // This calls the destructor without calling operator delete, allocator owns the memory.
      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_.fetch_sub(1, memory_order_relaxed);
    };
    num_active_threads_ = num_threads_;
    for (int i = 0; i < num_threads_; i++)
      workers_->Schedule(aligner_func);
  }

  Status SnapSingleExecutor::ok() const {
    return compute_status_;
  }
} // namespace tensorflow {
