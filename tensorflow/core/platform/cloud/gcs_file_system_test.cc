/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class FakeAuthProvider : public AuthProvider {
 public:
  Status GetToken(string* token) override {
    *token = "fake_token";
    return Status::OK();
  }
};

TEST(GcsFileSystemTest, NewRandomAccessFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-5\n",
           "012345"),
       new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-11\n",
           "6789")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  RandomAccessFile* file_ptr;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", &file_ptr));
  std::unique_ptr<RandomAccessFile> file(file_ptr);

  char scratch[6];
  StringPiece result;

  // Read the first chunk.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("012345", result);

  // Read the second chunk.
  EXPECT_EQ(
      errors::Code::OUT_OF_RANGE,
      file->Read(sizeof(scratch), sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("6789", result);
}

TEST(GcsFileSystemTest, NewWritableFile) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
      "uploadType=media&name=path/writeable.txt\n"
      "Auth Token: fake_token\n"
      "Post body: content1,content2\n",
      "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  WritableFile* file_ptr;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file_ptr));
  std::unique_ptr<WritableFile> file(file_ptr);

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewAppendableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/path/appendable.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-1048575\n",
           "content1,"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=media&name=path/appendable.txt\n"
           "Auth Token: fake_token\n"
           "Post body: content1,content2\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  WritableFile* file_ptr;
  TF_EXPECT_OK(
      fs.NewAppendableFile("gs://bucket/path/appendable.txt", &file_ptr));
  std::unique_ptr<WritableFile> file(file_ptr);

  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const string content = "file content";
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"", content.size(), "\"}")),
       new FakeHttpRequest(
           strings::StrCat(
               "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
               "Auth Token: fake_token\n"
               "Range: 0-",
               content.size() - 1, "\n"),
           content)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  ReadOnlyMemoryRegion* region_ptr;
  TF_EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile(
      "gs://bucket/random_access.txt", &region_ptr));
  std::unique_ptr<ReadOnlyMemoryRegion> region(region_ptr);

  EXPECT_EQ(content, StringPiece(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST(GcsFileSystemTest, FileExists) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path/file1.txt?fields=size\n"
           "Auth Token: fake_token\n",
           "{\"size\": \"100\"}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path/file2.txt?fields=size\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  EXPECT_TRUE(fs.FileExists("gs://bucket/path/file1.txt"));
  EXPECT_FALSE(fs.FileExists("gs://bucket/path/file2.txt"));
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "prefix=path/&fields=items\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(3, children.size());
  EXPECT_EQ("gs://bucket/path/file1.txt", children[0]);
  EXPECT_EQ("gs://bucket/path/subpath/file2.txt", children[1]);
  EXPECT_EQ("gs://bucket/path/file3.txt", children[2]);
}

TEST(GcsFileSystemTest, GetChildren_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "prefix=path/&fields=items\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, DeleteFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path/file1.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  TF_EXPECT_OK(fs.DeleteFile("gs://bucket/path/file1.txt"));
}

TEST(GcsFileSystemTest, DeleteDir_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "prefix=path/&fields=items\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/"));
}

TEST(GcsFileSystemTest, DeleteDir_NonEmpty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "prefix=path/&fields=items\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  EXPECT_FALSE(fs.DeleteDir("gs://bucket/path/").ok());
}

TEST(GcsFileSystemTest, GetFileSize) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size\n"
      "Auth Token: fake_token\n",
      strings::StrCat("{\"size\": \"1010\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://bucket/file.txt", &size));
  EXPECT_EQ(1010, size);
}

TEST(GcsFileSystemTest, RenameFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/src.txt"
           "/rewriteTo/b/bucket/o/dst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/src.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)));

  TF_EXPECT_OK(fs.RenameFile("gs://bucket/src.txt", "gs://bucket/dst.txt"));
}

}  // namespace
}  // namespace tensorflow
