#pragma once
#include <cstddef>
#include <string.h>
#include <memory>
#include "tensorflow/core/user_ops/dna-align/data.h"

namespace tensorflow {
    class Buffer : public Data {
    public:
        Buffer(std::size_t total_size);
        void WriteBuffer(const char* content, std::size_t size_of_content);
        virtual const char* data() const override;
        virtual std::size_t size() const override;
        const std::size_t total_size() const;

    private:
        std::size_t valid_size_;
        const std::size_t total_size_;
        std::unique_ptr<char> buf_;
        char* curr_position_;
    };
} // namespace tensorflow {
