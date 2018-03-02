
#include "tensorflow/contrib/persona/kernels/agd-ops/agd_reference_genome.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"

namespace tensorflow {

  using namespace std;

  Status AGDReferenceGenome::Create(AGDReferenceGenome* genome, const std::vector<const char*>& base_chunks, const vector<uint32_t>& base_lens, 
      const std::vector<const char*>& meta_chunks, const vector<uint32_t>& meta_lens) {

    CHECK_EQ(base_chunks.size(), meta_chunks.size());
    RecordParser parser;
    Buffer meta_buffer;
    uint64_t ordinal;
    uint32_t num_records;
    string record_id;

    genome->contigs_.reserve(base_chunks.size());
    genome->contig_names_.reserve(base_chunks.size());
    genome->contig_lens_.reserve(base_chunks.size());
    genome->contig_bufs_.resize(base_chunks.size());
    for (uint32_t i = 0; i < base_chunks.size(); i++) {
      TF_RETURN_IF_ERROR(parser.ParseNew(meta_chunks[i], meta_lens[i],
            false/*verify*/, &meta_buffer, &ordinal, &num_records, record_id, false/*unpack*/));
      CHECK_EQ(num_records, 1);

      // don't _need_ to use rec reader here, but it's easier
      AGDRecordReader metareader(meta_buffer.data(), num_records);
      const char* data;
      size_t len;
      TF_RETURN_IF_ERROR(metareader.GetNextRecord(&data, &len));
      genome->contig_names_.push_back(string(data, len));
      LOG(INFO) << "chunk name: " << string(data, len);
      
      TF_RETURN_IF_ERROR(parser.ParseNew(base_chunks[i], base_lens[i],
            false/*verify*/, &genome->contig_bufs_[i], &ordinal, &num_records, record_id, true/*unpack*/));
      CHECK_EQ(num_records, 1);
      
      AGDRecordReader contigreader(genome->contig_bufs_[i].data(), num_records);
      TF_RETURN_IF_ERROR(contigreader.GetNextRecord(&data, &len));
      genome->contigs_.push_back(data);
      genome->contig_lens_.push_back(len);

      LOG(INFO) << "contig length: " << len;
    }

    return Status::OK();
  }
  
  const char* AGDReferenceGenome::GetSubstring(uint32_t contig, uint32_t position) {
    CHECK_LT(contig, contigs_.size());
    CHECK_LT(position, contig_lens_[contig]);
    return &contigs_[contig][position];
  }

  uint32_t AGDReferenceGenome::GetContigLength(uint32_t contig) {
    CHECK_LT(contig, contigs_.size());
    return contig_lens_[contig];
  }

}
