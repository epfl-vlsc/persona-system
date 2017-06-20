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

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include "tensorflow/core/lib/core/errors.h"
#include "buffer.h"

namespace tensorflow {
namespace format {
  typedef uint16_t RelativeIndex;
  const uint16_t MAX_INDEX_SIZE = UINT16_MAX;

  const uint8_t current_major = 0;
  const uint8_t current_minor = 1;

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

    FileHeader() : version_major(current_major), version_minor(current_minor),
                   segment_start(sizeof(FileHeader)) {}
  };

  enum CompressionType {
    UNCOMPRESSED = 0,
    BZIP2 = 1,
    GZIP = 2
  };

  enum RecordType {
    TEXT = 0,
    STRUCTURED = 1,
    COMPACTED_BASES = 2
  };

  enum BaseAlphabet {
    A = 0,
    C = 1,
    T = 2,
    G = 3,
    N = 4,
    END = 7
  };


  /*struct __attribute__((packed)) AlignmentResult {
    uint16_t flag_ = 0; // ResultFlag
    int mapq_ = 0;
    int64_t location_ = 0; // POS field in SAM format
    int64_t next_location_ = 0; // the relative genomeLocation of the next segment (read)
    int64_t template_length_ = 0; // signed distance from leftmost mapped base to rightmost mapped base

    void convertFromSNAP(const SingleAlignmentResult &result, const int flag);
    std::string ToString() const {
      return string("flag: ") + std::to_string(flag_) + " mapq: " + std::to_string(mapq_) +
        " location: " + std::to_string(location_) + " nextLocation: " + std::to_string(next_location_) +
        "templateLen: " + std::to_string(template_length_);
    }
   };*/

  struct __attribute__((packed)) BinaryBases {
    BinaryBases() : bases(0) {};

    Status append(Buffer &output, std::size_t *num_bases) const;

    Status getBase(const std::size_t position, char* base) const;

    Status setBase(const char base, std::size_t position);

    Status terminate(std::size_t position);

    uint64_t bases;

    static const std::size_t compression = 21; //sizeof(uint64_t) * 2; // 4 bits = 2 per byte
    static const std::size_t base_width = 3;

  protected:


    Status setBaseAtPosition(const BaseAlphabet base, const std::size_t position);
  };

  Status append(const BinaryBases *bases, const std::size_t record_size_in_bytes, Buffer &data, Buffer &lengths);

  Status IntoBases(const char *fastq_base, const std::size_t fastq_base_size, std::vector<BinaryBases> &bases);

} // namespace format
} // namespace tensorflow
