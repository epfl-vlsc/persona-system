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

#ifndef TENSORFLOW_CORE_USER_OPS_FORMAT_H_
#define TENSORFLOW_CORE_USER_OPS_FORMAT_H_

#include <cstdint>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace format {
  struct __attribute__((packed)) RecordTable {
    uint8_t relative_index[];
  };

  struct __attribute__((packed)) FileHeader {
    uint8_t version_major;
    uint8_t version_minor;
    uint8_t record_type;
    uint8_t compression_type;
    uint16_t segment_start;
    uint16_t _padding;
    uint64_t first_ordinal;
    uint64_t last_ordinal;
    char string_id[32]; // FIXME: just make it static for now
  };

  enum CompressionType {
    UNCOMPRESSED = 0,
    BZIP2 = 1,
    GZIP = 2
  };

  enum RecordType {
    BASES = 0,
    QUALITIES = 1,
    COMMENTS = 2
  };

  enum BaseAlphabet {
    A = 0,
    C = 1,
    T = 2,
    G = 3,
    N = 4,
    END = 7
  };

  struct __attribute__((packed)) BinaryBases {
    BinaryBases() : bases(0) {};

    Status getBase(const std::size_t position, char* base) const;

    uint64_t bases;

    static const std::size_t compression = 21; //sizeof(uint64_t) * 2; // 4 bits = 2 per byte

  protected:
    static const std::size_t base_width = 3;
  };

  struct __attribute__((packed)) BinaryBaseRecord {
    /*
    static
    std::size_t
    intoBases(const char *fastq_base, const std::size_t fastq_base_size, std::vector<uint64_t> &bases);
    */
    Status
      toString(const std::size_t record_size_in_bytes, std::string *output) const;

    BinaryBases bases[]; // relative length stored in the RecordTable.relative_index
  };
  
} // namespace format
} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FORMAT_H_
