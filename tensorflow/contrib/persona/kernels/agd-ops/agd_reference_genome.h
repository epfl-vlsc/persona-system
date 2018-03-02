/*
 * agd_reference_genome.h
 * Stuart Byma
 * 
 * Parses and stores a reference genome in memory from an AGD ref genome dataset
 * Provides methods to get substrings etc. 
 */

#pragma once

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"

namespace tensorflow {

class AGDReferenceGenome {

  public:
    AGDReferenceGenome() {}
    ~AGDReferenceGenome() {}

    // loads genome from base and meta chunks into `genome`
    static Status Create(AGDReferenceGenome* genome,  const std::vector<const char*>& base_chunks, const std::vector<uint32_t>& base_lens, 
      const std::vector<const char*>& meta_chunks, const std::vector<uint32_t>& meta_lens);

    // return pointer into contig at position, asserts on out of bounds
    const char* GetSubstring(uint32_t contig, uint32_t position);

    // return length of give contig
    uint32_t GetContigLength(uint32_t contig);

    // return number of contigs
    uint32_t GetNumContigs();

  private:
    AGDReferenceGenome(const AGDReferenceGenome& other); // non construction-copyable
    AGDReferenceGenome& operator=(const AGDReferenceGenome&); // non copyable

    std::vector<string> contig_names_;
    std::vector<Buffer> contig_bufs_; // owns all buf data, which is why we are non copyable (~GBs of data)
    std::vector<const char*> contigs_; // just points into the buffers
    std::vector<uint32_t> contig_lens_;
};

} // namespace tf
