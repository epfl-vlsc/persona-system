#ifndef TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_

#include "tensorflow/core/lib/core/errors.h"

#include <vector>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace tensorflow {

template <typename T>
Status decompressSegment(const char* segment,
                         const std::size_t segment_size,
                         std::vector<char> &output);


auto const decompressBZIP2 = &decompressSegment<boost::iostreams::bzip2_decompressor>;
auto const decompressGZIP = &decompressSegment<boost::iostreams::gzip_decompressor>;


} // namespace tensorflow

#endif
