#include <cstddef>
#include <string.h>
#include "tensorflow/core/user_ops/dna-align/data.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"


namespace tensorflow {

  using namespace std;

    Buffer::Buffer(size_t total_size) : total_size_(total_size) {
        valid_size_ = 0;
        buf_.reset(new char[total_size]);
        curr_position_ = buf_.get();
    }

    void Buffer::WriteBuffer(const char* content, size_t size_of_content) {
        memcpy(curr_position_, content, size_of_content);
        curr_position_ += size_of_content;
        valid_size_ += size_of_content;
    }

    const char* Buffer::data() const {
        return buf_.get();
    }

    size_t Buffer::size() const {
        return valid_size_;
    }

    const size_t Buffer::total_size() const {
        return total_size_;
    }
} // namespace tensorflow {
