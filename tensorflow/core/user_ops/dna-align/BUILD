# user ops
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")

tf_proto_library_cc(
    name = "snap_protos",
    srcs = ["snap_proto.proto"],
    cc_libs = ["//tensorflow/core:protos_all_cc"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "snap_lib",
    srcs = glob(["snap/SNAPLib/*.cpp"]),
    hdrs = glob(["snap/SNAPLib/*.h"]),
    includes = ["snap/SNAPLib"],
    linkopts = [
        "-lm", "-lrt", "-lz"
    ],
    visibility = ["//visibility:public"],
    copts = ["-O3", "-pthread", "-MMD", "-msse", "-Wno-format", "-std=c++11"]
)

cc_library(
    name = "snap_tensor_decoders",
    srcs = ["snap_read_decode.cc",
            "snap_results_decode.cc"],
    hdrs = ["snap_read_decode.h",
            "snap_results_decode.h"],
    includes = ["."],
    copts = tf_copts(),
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

cc_library(
    name = "snap_align",
    hdrs = ["SnapAlignerWrapper.h", "genome_index_resource.h"],
    includes = ["."],
    srcs = glob(["*.cc"], 
           exclude = [
                      "sam_writer_async_op.cc", 
                      "snap_read_decode.cc",
                      "snap_results_decode.cc",
                      ],
               ),
    copts = tf_copts(),
    #linkopts = [
        #"-Wl,-Bsymbolic",
        #"-lm"
    #],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework",
            ":snap_protos_cc",
            "//tensorflow/core/user_ops/lttng:trace",
            "//tensorflow/core/user_ops/object-pool:object_pool",
            "//tensorflow/core/user_ops/dense-format:dense_format",
            ":snap_lib"],
    alwayslink=1,
    linkopts = [
             "-lboost_system", "-lboost_timer"
    ]
)

cc_library(
    name = "sam_writer_async",
    srcs = ["sam_writer_async_op.cc"],
    includes = ["."],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework", "//tensorflow/core/kernels:writer_async_base",
            ":snap_protos_cc", ":snap_tensor_decoders", ":snap_lib"],
    alwayslink = 1,
)
