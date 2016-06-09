#include "ref_pool.h"
#include <utility>

namespace tensorflow {
using namespace std;

void ReferencePool::AddResource(const string &container, const string &name)
{
  auto a = make_pair(container, name);
  objects_.push_back(move(a));
}

void ReferencePool::ReturnResource(const string &container, const string &name)
{
  // TODO lock and stuff
  mutex_lock l(objects_mu_);
  if (run_) {
    AddResource(container, name);
    objects_cv_.notify_one();
  }
}

ReferencePool::~ReferencePool()
{
  mutex_lock l(objects_mu_);
  objects_.clear();
  run_ = false;
  objects_cv_.notify_all();
}

void ReferencePool::GetResource(string &container, string &name)
{
  mutex_lock l(objects_mu_);
  // while instead of for in case multiple are woken up
  while (objects_.empty() && run_) {
    objects_cv_.wait(l, [this]() -> bool {
        return !objects_.empty() && run_;
      });
  }

  if (run_) {
    // guaranteed that there is at least one item in the queue now
    auto const &s = objects_.front();
    objects_.pop_front();
    container = move(s.first);
    name = move(s.second);
  }
}

string ReferencePool::DebugString()
{
  static const string s = "a reference pool";
  return s;
}

} // namespace tensorflow {
