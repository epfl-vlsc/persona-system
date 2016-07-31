#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include "data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
    class Buffer : public Data {
    private:
      std::vector<char> buf_;
      volatile bool data_ready_;
    public:
        Status WriteBuffer(const char* content, std::size_t content_size);
        Status AppendBuffer(const char* content, std::size_t content_size);
        Status AppendBufferDouble(const char* content, std::size_t content_size);
        decltype(buf_)& get();

        // this method will spin and block for this thread to be empty
        // very simple for now. use more fancy stuff when available
        decltype(buf_)& get_when_ready();

        void set_ready();
        void reset();
        virtual const char* data() const override;
        virtual std::size_t size() const override;
    };
} // namespace tensorflow {
