//
// Created by Stuart Byma on 17/04/17.
//

#include "tensorflow/contrib/persona/kernels/snap-align/paired_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  SnapPairedExecutor::SnapPairedExecutor(Env *env, GenomeIndex *index, PairedAlignerOptions *options,
                                         int num_threads, int capacity) : index_(index),
                                                                          options_(options),
                                                                          num_threads_(num_threads),
                                                                          capacity_(capacity) {
    genome_ = index_->getGenome();
    // create a threadpool to execute stuff
    workers_.reset(new thread::ThreadPool(env, "SnapPaired", num_threads_));
    request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<ReadResource>>>(capacity));
    auto s = snap_wrapper::init();
    if (!s.ok()) {
      compute_status_ = s;
      LOG(ERROR) << "snap_wrapper::init() returned an error in paired aligner init";
    } else {
      init_workers();
    }
  }

  SnapPairedExecutor::~SnapPairedExecutor() {
    if (!run_) {
      LOG(ERROR) << "Unable to safely wait in ~SnapAlignPairedOp for all threads. run_ was toggled to false\n";
    }
    run_ = false;
    request_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Status SnapPairedExecutor::EnqueueChunk(std::shared_ptr<ResourceContainer<ReadResource> > chunk) {
    if (!compute_status_.ok()) return compute_status_;
    if (!request_queue_->push(chunk))
      return Internal("Paired executor failed to push to request queue");
    else
      return Status::OK();
  }

  void SnapPairedExecutor::init_workers() {

    auto aligner_func = [this]() {
      snap_wrapper::PairedAligner aligner(options_, index_);

      PairedAlignmentResult primaryResult;
      vector<AlignmentResultBuilder> result_builders;
      array<Read, 2> snap_read;
      LandauVishkinWithCigar lvc;

      vector<BufferPair*> result_bufs;
      ReadResource* subchunk_resource = nullptr;
      PairedAlignmentResult *secondary_results;
      SingleAlignmentResult *secondary_single_results;
      Status io_chunk_status, subchunk_status;
      bool useless[2];
      int num_secondary_results, num_secondary_single_results_first, num_secondary_single_results_second, num_secondary;
      size_t num_columns;

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<ReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        ScopeDropIfEqual<decltype(reads_container)> scope_dropper(*request_queue_, reads_container);

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {

          num_columns = result_bufs.size();
          num_secondary = num_columns - 1;

          // only ever expand result builders, to avoid reallocation
          if (num_columns > result_builders.size()) {
            result_builders.resize(num_columns);
          }

          // re-initialize the result builders with the result buffer of the corresponding column
          for (decltype(num_columns) i = 0; i < num_columns; i++) {
            result_builders[i].SetBufferPair(result_bufs[i]);
          }

          subchunk_status = Status::OK();
          while (subchunk_status.ok()) {
            // assume reads are successive.
            size_t read_idx;
            for (read_idx = 0; read_idx < 2; ++read_idx) {
              auto &sread = snap_read[read_idx];
              subchunk_status = subchunk_resource->get_next_record(sread);
              if (subchunk_status.ok()) {
                sread.clip(options_->clipping);
                useless[read_idx] = sread.getDataLength() < options_->minReadLength && sread.countOfNs() > options_->maxDist;
              } else {
                break;
              }
            }

            if (!subchunk_status.ok()) {
              if (!(IsResourceExhausted(subchunk_status) && read_idx == 0)) {
                LOG(ERROR) << "Subchunk is exhausted on an odd number of records";
                subchunk_status = Internal("Unable to get 2 records for snap read. Must have an even number in the subchunking!");
              }
              break;
            }

            if (useless[0] || useless[1]) {
              // TODO change to the writePairs code for the new readWriter
              // we just filter these out for now
              for (size_t i = 0; i < 2; ++i) {
                primaryResult.status[i] = AlignmentResult::NotFound;
                primaryResult.location[i] = InvalidGenomeLocation;
                primaryResult.mapq[i] = 0;
                primaryResult.direction[i] = FORWARD;
                subchunk_status = aligner.writeResult(snap_read, primaryResult, result_builders[0], false);
                // fill in blanks for secondaries
                for (decltype(num_columns) i = 1; i < num_columns; i++) {
                  result_builders[i].AppendEmpty();
                }
              }
              continue;
            }

            aligner.align(snap_read, primaryResult, num_secondary, &secondary_results, &num_secondary_results,
                          &secondary_single_results, &num_secondary_single_results_first,
                          &num_secondary_single_results_second);

            // TODO do we need a check here?
            subchunk_status = aligner.writeResult(snap_read, primaryResult, result_builders[0], false);

            int i = 0;
            if (num_secondary_results > 0) {
              while (subchunk_status.ok() && i < num_secondary_results) {
                subchunk_status = aligner.writeResult(snap_read, secondary_results[i], result_builders[i+1], true);
                i++;
              }
              // FIXME should this be else, or else if?
            } else if (num_secondary_single_results_first > 0 || num_secondary_single_results_second > 0) {
              while (subchunk_status.ok() && i < num_secondary_single_results_first) {
                subchunk_status = snap_wrapper::WriteSingleResult(snap_read[0], secondary_single_results[i],
                                                                  result_builders[i+1], index_->getGenome(), &lvc, true, options_->useM);
                i++;
              }
              // fill in blanks
              i = num_secondary_single_results_first;
              while (i < num_secondary) {
                result_builders[i + 1].AppendEmpty();
                i++;
              }
              i = 0;
              // neither single secondary results should have more than max_secondary_
              while (subchunk_status.ok() && i < num_secondary_single_results_second) {
                subchunk_status = snap_wrapper::WriteSingleResult(snap_read[1],
                                                                  secondary_single_results[i+num_secondary_single_results_first],
                                                                  result_builders[i+1], index_->getGenome(), &lvc, true, options_->useM);
                i++;
              }
              // fill in the gaps
              i = num_secondary_single_results_second;
              while (i < num_secondary) {
                result_builders[i + 1].AppendEmpty();
                i++;
              }
            }
          } // subchunk loop

          if (!IsResourceExhausted(subchunk_status)) {
            compute_status_ = subchunk_status;
            run_ = false;
            break;
          }

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        } // io chunk loop
      }

      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_.fetch_sub(1, memory_order_relaxed);
    };

    num_active_threads_ = num_threads_;
    for (uint_fast32_t i = 0; i < num_threads_; i++)
      workers_->Schedule(aligner_func);
  }

  Status SnapPairedExecutor::ok() const {
    return compute_status_;
  }

} // namespace tensorflow {
