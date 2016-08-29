#include "compression.h"
#include "util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using namespace std;
using namespace errors;

namespace {

static const int ENABLE_ZLIB_GZIP = 32;
static const int ENABLE_ZLIB_GZIP_COMPRESS = 16;
static const int window_bits = 15;
static const size_t extend_length = 1024 * 1024 * 8; // 2 Mb
static const size_t reserve_factor = 3;

Status resize_output(z_stream &strm, vector<char> &output, size_t extend_len) {
  auto s = Status::OK();

  if (strm.avail_out == 0) {
    auto new_cap = output.capacity() + extend_len;
    safe_reserve(output, new_cap);
    output.resize(output.capacity());
    if (output.capacity() < new_cap) {
      s = Internal("Unable to reserve more capacity in a buffer");
    } else {
      strm.next_out = reinterpret_cast<unsigned char*>(&output[strm.total_out]);
      // should never be negative
      strm.avail_out = output.capacity() - strm.total_out;
    }
  }

  return s;
}

}

Status decompressGZIP(const char* segment,
                      const std::size_t segment_size,
                      Buffer *output)
{
  return Unimplemented("implemented gzip decom for Buffer* !!!");
}

Status decompressGZIP(const char* segment,
                      const size_t segment_size,
                      vector<char> &output)
{
  // TODO this only supports decompress write, not appending
  // this is an easy change to make, but requires some more "math"
  output.clear(); // just to be sure, in case the caller didn't do it
  static const int init_flags = window_bits | ENABLE_ZLIB_GZIP;
  z_stream strm = {0};
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(segment));
  strm.avail_in = segment_size;
  safe_reserve(output, segment_size * reserve_factor, 16 * 1024 * 1024);

  int status = inflateInit2(&strm, init_flags);
  if (status != Z_OK) {
    return Internal("inflateInit2 failed with error: ", status);
  }

  auto s = Status::OK();

  // First, try to decompress as much as possible in a single step
  strm.avail_out = output.capacity();
  strm.next_out = reinterpret_cast<unsigned char*>(&output[0]);
  output.resize(output.capacity());
  status = inflate(&strm, Z_FINISH);
  // TODO do we need to call inflate in while lopp while status == Z_OK (like deflate)?

  if (status != Z_STREAM_END) {
    if (status == Z_OK || status == Z_BUF_ERROR) {
      // Do normal decompression because we couldn't do it in one shot
      s = resize_output(strm, output, extend_length);
      while (status != Z_STREAM_END && s.ok() && strm.total_in < segment_size) {
        status = inflate(&strm, Z_NO_FLUSH);
        switch (status) {
        case Z_BUF_ERROR:
        case Z_OK:
          s = resize_output(strm, output, extend_length);
        case Z_STREAM_END:
          break;
        default: // an error
          s = Internal("inflate(Z_NO_FLUSH) returned code ", status, " with message '", strm.msg == NULL ? "" : strm.msg, "'");
          break;
        }
      }
    } else {
      s = Internal("inflate(Z_FINISH) return code ", status, " with message '", strm.msg == NULL ? "" : strm.msg, "'");
    }
  }

  auto total_out = strm.total_out;

  status = inflateEnd(&strm);
  if (status != Z_OK) { // s.ok() status to make sure we don't override non-inflateEnd error
    if (s.ok()) {
      s = Internal("inflateEnd() didn't receive Z_OK. Got: ", status);
    }
    output.clear();
  } else {
    output.resize(total_out);
  }
  return s;
}

Status compressGZIP(const char* segment,
                    const size_t segment_size,
                    vector<char> &output)
{
  output.clear(); // TODO for now we just overwrite. No appending support yet.

  // Some setup for zlib
  z_stream strm = {0};
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  strm.next_in = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(segment));
  strm.avail_in = segment_size;

  // TODO I think this needs to be deflateInit2, to use GZIP
  int status = deflateInit2(&strm, Z_DEFAULT_COMPRESSION,
                            Z_DEFLATED, window_bits | ENABLE_ZLIB_GZIP_COMPRESS,
                            9, // higher memory, better speed
                            Z_DEFAULT_STRATEGY);

  if (status != Z_OK) {
    return Internal("deflateInit() didn't return Z_OK. Return ", status, " with 2nd param ", Z_DEFAULT_COMPRESSION);
  }

  auto s = Status::OK();

  auto deflate_size = deflateBound(&strm, segment_size);
  output.resize(deflate_size);

  strm.next_out = reinterpret_cast<unsigned char*>(&output[0]);
  strm.avail_out = output.size();

  // Try to do it in one shot, if possible
  status = deflate(&strm, Z_FINISH);
  while (status == Z_OK && s.ok()) {
    s = resize_output(strm, output, extend_length);
    status = deflate(&strm, Z_FINISH);
  }

  if (status != Z_STREAM_END) {
    s = Internal("deflate(Z_FINISH) returned non-OK, non-END error: ", status);
  }

  auto total_out = strm.total_out; // save it here because deflate nukes this value
  status = deflateEnd(&strm);
  if (status != Z_OK) {
    if (s.ok()) {
      s = Internal("deflateEnd() didn't receive Z_OK. Got: ", status);
    }
    output.clear();
  } else {
    output.resize(total_out);
  }

  return s;
}

  // reinitializes the stream
  Status AppendingGZIPCompressor::init() {
    // TODO this should all be done in the compressor
    stream_ = {0};
    // Not sure if this is necessary
    stream_.zalloc = Z_NULL;
    stream_.zfree = Z_NULL;
    stream_.opaque = Z_NULL;
    done_ = false;
    output_.clear();
    safe_reserve(output_, 1);

    int status = deflateInit2(&stream_, Z_DEFAULT_COMPRESSION,
                              Z_DEFLATED, window_bits | ENABLE_ZLIB_GZIP_COMPRESS,
                              9, // higher memory, better speed
                              Z_DEFAULT_STRATEGY);
    if (status != Z_OK) {
        return Internal("deflateInit() didn't return Z_OK. Return ", status, " with 2nd param ", Z_DEFAULT_COMPRESSION);
    }
  }

  Status AppendingGZIPCompressor::appendGZIP(const char* segment,
                                             const size_t segment_size) {
    if (done_) {
      return Unavailable("appendGZIP is already finished. Must call init()");
    }

    stream_.next_in = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(segment));
    stream_.avail_in = segment_size;

    int status;
    auto init_size = output_.size();
    output_.resize(output_.capacity());

    stream_.avail_out = output_.size() - init_size;
    stream_.next_out = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(&output_[init_size]));
    do {
      status = deflate(&stream_, Z_NO_FLUSH); // TODO maybe this has to be flushed
      switch (status) {
      default:
        return Internal("deflate(Z_NO_FLUSH) return status ", status);
        break;
      case Z_OK:
        break;
      }
    } while (stream_.avail_out == 0);

    return Status::OK();
  }

  // closes the stream
  Status AppendingGZIPCompressor::finish() {
    auto status = deflateEnd(&stream_);
    if (status != Z_OK) {
      return Internal("deflateEnd() didn't receive Z_OK. Got: ", status);
    }
    return Status::OK();
  }

  AppendingGZIPCompressor::~AppendingGZIPCompressor()
  {
    finish(); // TODO maybe log here?
  }

  AppendingGZIPCompressor::AppendingGZIPCompressor(vector<char> &output) : output_(output)
  {
    
  }
} // namespace tensorflow
