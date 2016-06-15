#include <cstddef>
#include <string.h>
#include "tensorflow/core/user_ops/dna-align/data.h"

using namespace std;

namespace tensorflow {
    class Buffer : public Data {
    public:
        Buffer(size_t total_size);
        ~Buffer();
        void WriteBuffer(const char* content, size_t size_of_content);
        virtual const char* data() const override;
        virtual size_t size() const override;
        const size_t total_size() const;

    private:
        size_t valid_size_;
        const size_t total_size_;
        const char* buf_;
        char* curr_position_;
    };
} // namespace tensorflow {
