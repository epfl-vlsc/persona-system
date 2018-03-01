
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")

# compile swps3 into a lib with just the things we need (SSE, scalar)
cc_library(
    name = "swps3_lib",
    srcs = [
            "DynProgr_sse_short.c",
            "DynProgr_sse_byte.c",
            "DynProgr_scalar.c",
            "matrix.c",
            "fasta.c",
            "debug.c",
            ],
    hdrs = ["swps3.h",
            "DynProgr_sse_short.h",
            "DynProgr_sse_byte.h",
            "DynProgr_scalar.h",
            "matrix.h",
            "debug.h",
            "fasta.h",
            ],
    linkopts = [
        "-msse3"
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    copts = ["-msse3"]
)

