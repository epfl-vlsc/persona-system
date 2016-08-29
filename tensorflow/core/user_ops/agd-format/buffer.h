#pragma once
#include <cstddef>
#include <memory>
#include "data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "buffer_list.h"

namespace tensorflow {
  class BufferList;

    class Buffer : public Data {
    private:
      std::unique_ptr<char[]> buf_{nullptr};
      std::size_t size_ = 0, allocation_ = 0, extend_extra_ = 0;

      void set_buffer_list_parent(BufferList *buffer_list);
      BufferList *parent_ = nullptr;

      friend class BufferList;
    public:

      Buffer(decltype(size_) initial_size = 64 * 1024,
             decltype(size_) extend_extra = 8 * 1024 * 1024);

        Status WriteBuffer(const char* content, std::size_t content_size);
        Status AppendBuffer(const char* content, std::size_t content_size);

        // ensures that the total capacity is at least `capacity`
        void reserve(std::size_t capacity);

        // resizes the actual size to to total_size
        // returns an error if total_size > allocation_
        // should call `reserve` first to be safe
        inline Status resize(decltype(size_) total_size);

        // extends the current allocation by `extend_size`
        // does not affect size_
        inline void extend_allocation(decltype(size_) extend_size);

        // extends the current size by extend_size
        // returns an error if size_+extend_size > allocation
        // should call `extend_allocation` first to be safe
        inline Status extend_size(decltype(size_) extend_size);

        char& operator[](size_t idx) const;

        void set_ready();
        void reset();
        virtual const char* data() const override;
        virtual std::size_t size() const override;
    };
} // namespace tensorflow {
