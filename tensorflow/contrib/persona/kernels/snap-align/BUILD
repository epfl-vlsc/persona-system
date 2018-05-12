# user ops
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")

# Compilation fo the SNAP library
cc_library(
    name = "snap_lib",
    #srcs = glob(["libsnap.a"]),
    srcs = glob(["snap/SNAPLib/*.cpp"]),
    hdrs = glob(["snap/SNAPLib/*.h"]),
    includes = ["snap/SNAPLib"],
    linkopts = [
        "-lm", 
        #"-lrt", 
        "-lz"
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"]
)

cc_library(
    name = "snap_align",
    hdrs = ["SnapAlignerWrapper.h", "single_executor.h"],
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
            "//tensorflow/core:framework_headers_lib",
            #":snap_protos_cc",
            ":snap_lib",
            #"//tensorflow/contrib/persona/kernels/lttng:trace",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            "//tensorflow/contrib/persona/kernels/concurrent_queue:concurrent_queue"
            ],
    alwayslink=1,
    linkopts = [
             "-lboost_system", "-lboost_timer"
    ]
)
