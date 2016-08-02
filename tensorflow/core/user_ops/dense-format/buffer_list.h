#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include <memory>
#include "buffer.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
    class BufferList {
    private:
      std::vector<std::unique_ptr<Buffer>> buf_list_;

    public:
        Buffer* get_at(size_t index);
        void resize(size_t size);
        decltype(buf_list_)& get();
        void reset();
    };
} // namespace tensorflow {
