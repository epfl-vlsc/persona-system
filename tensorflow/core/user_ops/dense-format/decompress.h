/* Copyright 2015 Google Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ==============================================================================*/
#ifndef TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_

#include "tensorflow/core/lib/core/errors.h"

#include <vector>
#include <boost/iostreams/filter/bzip2.hpp>

namespace tensorflow {

template <typename T>
Status decompressSegment(const char* segment,
                         const std::size_t segment_size,
                         std::vector<char> &output);

auto const decompressBZIP2 = &decompressSegment<boost::iostreams::bzip2_decompressor>;

} // namespace tensorflow

#endif
