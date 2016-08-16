#pragma once

#include <chrono>
#include <sstream>
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;

  class ScopeTimer {
  public:
  ScopeTimer(WritableFile *trace_file) : trace_file_(trace_file),
      start_tick_(chrono::high_resolution_clock::now()),
      start_time_(chrono::system_clock::now()) {}

    ~ScopeTimer() {
      using namespace chrono;

      auto tick_diff = duration_cast<microseconds>(high_resolution_clock::now() - start_tick_);
      auto start_time_us = duration_cast<microseconds>(start_time_.time_since_epoch());
      ostringstream a;
      a << start_time_us.count() << "," << tick_diff.count() << "\n";
      auto s = trace_file_->Append(a.str());
      if (!s.ok()) {
        LOG(WARNING) << "Unable to append to trace file";
      }
    }
  private:
    WritableFile *trace_file_;

    chrono::high_resolution_clock::time_point start_tick_;
    chrono::system_clock::time_point start_time_;
  };
} // namespace tensorflow
