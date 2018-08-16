//
// Created by Stuart Byma on 17/05/18.
//
#pragma once

#include <chrono>
#include <atomic>
#include <thread>
#include <utility>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/multi_notification.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/cluster_seq.h"


namespace tensorflow {

  class Cluster; // forward declare, protein_cluster.h

  class AlignmentExecutor : public ResourceBase {


  public:

    typedef std::tuple<const ClusterSequence*, const ClusterSequence*, ProteinAligner::Alignment*, MultiNotification*> WorkItem;
    
    typedef std::tuple<Sequence, Cluster*, bool*, MultiNotification*> ClusterWorkItem;

    AlignmentExecutor(Env *env, int num_threads, int num_threads_align, int capacity);
    ~AlignmentExecutor();

    Status EnqueueAlignment(const WorkItem& item);
    
    Status EnqueueClusterEval(const ClusterWorkItem& item);

    // not, would be better if the Op received all this crap 
    // but im too lazy to override resourceOp compute
    void SetVars(CandidateMap* map, const AlignmentEnvironments* envs, 
        const Parameters* params ) { 
      if (!candidate_map_) {
        candidate_map_ = map; 
        params_ = params;
        envs_ = envs;
      }
    }
      
    string DebugString() override {
      return string("A AlignmentExecutor");
    }
    

    Status ok() const;

  private:
    volatile bool run_ = true;
    const AlignmentEnvironments* envs_ = nullptr;
    const Parameters* params_ = nullptr;
    CandidateMap* candidate_map_ = nullptr;

    std::atomic_uint_fast32_t num_active_threads_, id_{0}, total_alignments_{0};
    std::atomic_uint_fast32_t num_times_emptied_{0};
    mutex mu_;

    int num_threads_;
    int num_threads_align_;
    int capacity_;

    std::unique_ptr<ConcurrentQueue <WorkItem>> work_queue_;
    std::unique_ptr<ConcurrentQueue <ClusterWorkItem>> cluster_work_queue_;

    Status compute_status_ = Status::OK();
    std::unique_ptr<thread::ThreadPool> workers_;
    std::unique_ptr<thread::ThreadPool> eval_workers_;

    void init_workers();

  };
}
