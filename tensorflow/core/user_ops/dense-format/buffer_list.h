#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include "buffer.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
    class BufferList {
    public:
        Buffer* get_at(int index);
        void resize(size_t size);
        std::vector<Buffer>& get();
        void reset();

        ~BufferList();

    private:
        std::vector<Buffer> buf_list_;
    };
} // namespace tensorflow {
