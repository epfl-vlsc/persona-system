#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include "data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
    class Buffer : public Data {
    public:
        Status WriteBuffer(const char* content, std::size_t content_size);
        Status AppendBuffer(const char* content, std::size_t content_size);
        std::vector<char>& get();
        void reset();
        virtual const char* data() const override;
        virtual std::size_t size() const override;

    private:
        std::vector<char> buf_;
    };
} // namespace tensorflow {