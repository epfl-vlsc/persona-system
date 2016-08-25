#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include "data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "buffer_list.h"

namespace tensorflow {
  class BufferList;

    class Buffer : public Data {
    private:
      std::vector<char> buf_;

      void set_buffer_list_parent(BufferList *buffer_list);
      BufferList *parent_ = nullptr;

      friend class BufferList;
    public:

      Buffer() = default;

        Status WriteBuffer(const char* content, std::size_t content_size);
        Status AppendBuffer(const char* content, std::size_t content_size);
        Status AppendBufferDouble(const char* content, std::size_t content_size);
        decltype(buf_)& get();

        void set_ready();
        void reset();
        virtual const char* data() const override;
        virtual std::size_t size() const override;
    };
} // namespace tensorflow {
