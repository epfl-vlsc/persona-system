
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "bwa_lib",
    #srcs = glob(["libsnap.a"]),
    srcs = glob(["bwa/*.c"]),
    hdrs = glob(["bwa/*.h"]),
    includes = ["bwa"],
    linkopts = [
        "-lm", "-lpthread", "-lz", "-lrt"
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
    #linkopts = [
        #"-Wl,-Bsymbolic",
        #"-lm"
    #],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework",
            #":snap_protos_cc",
            ":bwa_lib",
            "//tensorflow/core/user_ops/lttng:trace",
            "//tensorflow/core/user_ops/object-pool:object_pool",
            "//tensorflow/core/user_ops/agd-format:agd_format"
            ],
    alwayslink=1,
    #linkopts = [
    #"-lboost_system", "-lboost_timer"
    #]
)
