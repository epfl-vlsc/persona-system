# TensorFlow external dependencies that can be loaded in WORKSPACE files.

# If TensorFlow is linked as a submodule, path_prefix is TensorFlow's directory
# within the workspace (e.g. "tensorflow/"), and tf_repo_name is the name of the
# local_repository rule (e.g. "@tf").
def tf_workspace(path_prefix = "", tf_repo_name = ""):
  native.new_http_archive(
    name = "eigen_archive",
    url = "https://bitbucket.org/eigen/eigen/get/f3a13643ac1f.tar.gz",
    sha256 = "a9266e60366cddb371a23d86b11a297eee86372a89ef4b38a3509012f9cc37ec",
    build_file = path_prefix + "eigen.BUILD",
  )

  native.git_repository(
    name = "re2",
    remote = "https://github.com/google/re2.git",
    commit = "791beff",
  )

  native.git_repository(
    name = "gemmlowp",
    remote = "https://github.com/google/gemmlowp.git",
    commit = "96d3acab46fbb03855ca22c2ee2bb9831ac8c83c",
  )

  native.new_http_archive(
    name = "farmhash_archive",
    url = "https://github.com/google/farmhash/archive/34c13ddfab0e35422f4c3979f360635a8c050260.zip",
    sha256 = "e3d37a59101f38fd58fb799ed404d630f0eee18bfc2a2433910977cc8fea9c28",
    build_file = path_prefix + "farmhash.BUILD",
  )

  native.bind(
    name = "farmhash",
    actual = "@farmhash//:farmhash",
  )

  native.git_repository(
    name = "highwayhash",
    remote = "https://github.com/google/highwayhash.git",
    commit = "be5edafc2e1a455768e260ccd68ae7317b6690ee",
    init_submodules = True,
  )

  native.new_http_archive(
    name = "jpeg_archive",
    url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
    sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
    build_file = path_prefix + "jpeg.BUILD",
  )

  native.new_http_archive(
    name = "png_archive",
    url = "https://github.com/glennrp/libpng/archive/v1.2.53.zip",
    sha256 = "c35bcc6387495ee6e757507a68ba036d38ad05b415c2553b3debe2a57647a692",
    build_file = path_prefix + "png.BUILD",
  )

  native.new_http_archive(
    name = "six_archive",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    build_file = path_prefix + "six.BUILD",
  )

  native.bind(
    name = "six",
    actual = "@six_archive//:six",
  )

  native.git_repository(
    name = "protobuf",
    remote = "https://github.com/google/protobuf",
    commit = "ed87c1fe2c6e1633cadb62cf54b2723b2b25c280",
  )

  native.new_http_archive(
    name = "gmock_archive",
    url = "https://archive.openswitch.net/gmock-1.7.0.zip",
    sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
    build_file = path_prefix + "gmock.BUILD",
  )

  native.bind(
    name = "gtest",
    actual = "@gmock_archive//:gtest",
  )

  native.bind(
    name = "gtest_main",
    actual = "@gmock_archive//:gtest_main",
  )

  native.bind(
      name = "python_headers",
      actual = tf_repo_name + "//util/python:python_headers",
  )

  # grpc expects //external:protobuf_clib and //external:protobuf_compiler
  # to point to the protobuf's compiler library.
  native.bind(
    name = "protobuf_clib",
    actual = "@protobuf//:protoc_lib",
  )

  native.bind(
    name = "protobuf_compiler",
    actual = "@protobuf//:protoc_lib",
  )

  native.git_repository(
    name = "grpc",
    commit = "2bc7d80",
    init_submodules = True,
    remote = "https://github.com/grpc/grpc.git",
  )

  # protobuf expects //external:grpc_cpp_plugin to point to grpc's
  # C++ plugin code generator.
  native.bind(
    name = "grpc_cpp_plugin",
    actual = "@grpc//:grpc_cpp_plugin",
  )

  native.bind(
    name = "grpc_lib",
    actual = "@grpc//:grpc++_unsecure",
  )

  native.new_git_repository(
    name = "jsoncpp_git",
    remote = "https://github.com/open-source-parsers/jsoncpp.git",
    commit = "11086dd6a7eba04289944367ca82cea71299ed70",
    build_file = path_prefix + "jsoncpp.BUILD",
  )

  native.bind(
    name = "jsoncpp",
    actual = "@jsoncpp_git//:jsoncpp",
  )

  native.new_git_repository(
    name = "boringssl_git",
    commit = "e72df93461c6d9d2b5698f10e16d3ab82f5adde3",
    remote = "https://boringssl.googlesource.com/boringssl",
    build_file = path_prefix + "boringssl.BUILD",
  )

  native.new_http_archive(
    name = "avro_archive",
    url = "http://www-us.apache.org/dist/avro/avro-1.8.0/cpp/avro-cpp-1.8.0.tar.gz",
    sha256 = "ec6e2ec957e95ca07f70cc25f02f5c416f47cb27bd987a6ec770dcbe72527368",
    build_file = path_prefix + "avro.BUILD",
  )

  native.new_http_archive(
    name = "boost_archive",
    url = "http://pilotfiber.dl.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz",
    sha256 = "a77c7cc660ec02704c6884fbb20c552d52d60a18f26573c9cee0788bf00ed7e6",
    build_file = path_prefix + "boost.BUILD",
  )

  native.new_http_archive(
    name = "bzip2_archive",
    url = "http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz",
    sha256 = "a2848f34fcd5d6cf47def00461fcb528a0484d8edef8208d6d2e2909dc61d9cd",
    build_file = path_prefix + "bzip2.BUILD",
  )

  native.new_http_archive(
    name = "zlib_archive",
    url = "http://zlib.net/zlib-1.2.8.tar.gz",
    sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
    build_file = path_prefix + "zlib.BUILD",
  )

