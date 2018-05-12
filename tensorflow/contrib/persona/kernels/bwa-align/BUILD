
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "bwa_lib",
    #srcs = glob(["libsnap.a"]),
    srcs = glob(["bwa/*.c"], exclude=["bwa/main*"]),
    hdrs = glob(["bwa/*.h"]),
    includes = ["bwa"],
    linkopts = [
        "-lm", "-lpthread", "-lz", 
        #"-lrt"
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    #copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"]
)


cc_library(
    name = "bwa_align",
    hdrs = ["bwa_wrapper.h"],
    includes = ["."],
    srcs = glob(["*.cc"]),
    copts = tf_copts() + ["-fexceptions"],
    nocopts = "-fno-exceptions",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework_headers_lib",
            ":bwa_lib",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/concurrent_queue:concurrent_queue",
            #"//tensorflow/contrib/persona/kernels/lttng:trace",
            "//third_party/eigen3",
            #"@protobuf//:protobuf",
            "@protobuf_archive//:protobuf",
            ],
    alwayslink=1,
)
