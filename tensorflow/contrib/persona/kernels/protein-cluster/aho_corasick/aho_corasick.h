
#pragma once

#include <iostream>
#include <queue>
#include <experimental/string_view>
#include <unordered_set>
#include <vector>

using std::cout;
using std::queue;
using std::string;
using std::experimental::string_view;
using std::unordered_set;
using std::vector;

// index based Aho Corasick, based on
// https://www.geeksforgeeks.org/aho-corasick-algorithm-pattern-searching/
// Idea is that indexing rather than pointers will play better with x86 mem
// system, and give better performance.
template <size_t AlphabetSize, bool DoNormalize>
class ACTrie {
 public:
  void Initialize(const unordered_set<string_view>& dict, uint32_t length_sum) {
    word_length_sum_ = length_sum + 1; // not sure if + 1 really needed
    out_func_.resize(word_length_sum_);
    std::fill(out_func_.begin(), out_func_.end(), 0);

    goto_func_.resize(word_length_sum_ * AlphabetSize, -1);
    goto_func_.SetWidth(word_length_sum_);
    std::fill(goto_func_.begin(), goto_func_.end(), -1);

    // build the trie, fill goto_func
    uint32_t states = 1;
    for (const auto& word : dict) {
      int current_state = 0;

      for (const auto& c : word) {
        uint32_t ch;
        if (DoNormalize) {
          ch = uint32_t(toupper(c) - 'A');
        } else {
          ch = uint32_t(c);
        }

        if (goto_func_(current_state, ch) == -1) {
          goto_func_(current_state, ch) = states++;
        }

        current_state = goto_func_(current_state, ch);
      }

      out_func_[current_state]++;
    }

    for (int ch = 0; ch < AlphabetSize; ++ch) {
      if (goto_func_(0, ch) == -1) {
        goto_func_(0, ch) = 0;
      }
    }

    num_states_ = states;

    failure_func_.resize(word_length_sum_);
    std::fill(failure_func_.begin(), failure_func_.end(), -1);

    queue<int> failure_queue;

    for (int ch = 0; ch < AlphabetSize; ++ch) {
      // All nodes of depth 1 have failure function value
      // as 0. For example, in above diagram we move to 0
      // from states 1 and 3.
      if (goto_func_(0, ch) != 0) {
        failure_func_[goto_func_(0, ch)] = 0;
        failure_queue.push(goto_func_(0, ch));
      }
    }

    while (failure_queue.size() != 0) {
      // Remove the front state from queue
      int state = failure_queue.front();
      failure_queue.pop();

      // For the removed state, find failure function for
      // all those characters for which goto function is
      // not defined.
      for (int ch = 0; ch < AlphabetSize; ++ch) {
        // If goto function is defined for character 'ch'
        // and 'state'
        if (goto_func_(state, ch) != -1) {
          // Find failure state of removed state
          int failure = failure_func_[state];

          // Find the deepest node labeled by proper
          // suffix of string from root to current
          // state.
          while (goto_func_(failure, ch) == -1)
            failure = failure_func_[failure];

          failure = goto_func_(failure, ch);
          failure_func_[goto_func_(state, ch)] = failure;

          // Merge output values
          out_func_[goto_func_(state, ch)] += out_func_[failure];

          // Insert the next level node (of Trie) in Queue
          failure_queue.push(goto_func_(state, ch));
        }
      }
    }
  }

  // search for instances of dict words in `text`
  // return the number of hits found
  uint32_t Search(string_view text) {
    if (word_length_sum_ == 0) {
      return 0;
    }
    uint32_t current_state = 0;
    uint32_t total_hits = 0;

    for (const auto& ch : text) {
      current_state = FindNextState(current_state, ch);

      total_hits += out_func_[current_state];
    }

    return total_hits;
  }

 private:
  inline uint32_t FindNextState(uint32_t current_state, char next_input) {
    int answer = current_state;
    uint16_t ch;
    if (DoNormalize) {
      ch = uint16_t(toupper(next_input) - 'A');
    } else {
      ch = uint16_t(next_input);
    }

    // If goto is not defined, use failure function
    while (goto_func_(answer, ch) == -1) answer = failure_func_[answer];

    return goto_func_(answer, ch);
  }
  // https://stackoverflow.com/a/34785963/2864407
  // Keep arrays contiguous for better mem system performance
  template <typename T>
  class Vector2D : public vector<T> {
   public:
    Vector2D(unsigned newWidth = 10) : std::vector<T>(), width_(newWidth) {}

    unsigned Width() { return width_; }
    void SetWidth(unsigned i) { width_ = i; }

    T& operator()(int x, int y) { return this->operator[](x + width_ * y); }

   protected:
    unsigned width_;
  };

  size_t word_length_sum_ = 0;
  // not a bitmap, just a counter of how many words matched here
  vector<uint16_t> out_func_;
  vector<int16_t> failure_func_;  // failure function
  Vector2D<int32_t> goto_func_;   // goto (state transition) function
  uint32_t num_states_ = 0;
};