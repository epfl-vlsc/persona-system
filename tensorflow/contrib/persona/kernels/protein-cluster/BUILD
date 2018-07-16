
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")


# compile swps3 into a lib with just the things we need (SSE, scalar)
cc_library(
    name = "swps3_lib",
    srcs = [
            "swps3/DynProgr_sse_short.c",
            "swps3/DynProgr_sse_byte.c",
            "swps3/DynProgr_sse_double.c",
            "swps3/DynProgr_scalar.c",
            "swps3/matrix.c",
            "swps3/extras.c",
            "swps3/fasta.c",
            "swps3/debug.c",
            "swps3/EstimatePam.c",
            "swps3/Page_size.c"
            ],
    hdrs = ["swps3/swps3.h",
            "swps3/DynProgr_sse_short.h",
            "swps3/DynProgr_sse_byte.h",
            "swps3/DynProgr_sse_double.h",
            "swps3/DynProgr_scalar.h",
            "swps3/extras.h",
            "swps3/matrix.h",
            "swps3/debug.h",
            "swps3/fasta.h",
            "swps3/EstimatePam.h",
            "swps3/Page_size.h"
            ],
    linkopts = [
        "-msse3"
    ],
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    copts = ["-msse3"]
)

# protein clustering ops
cc_library(
    name = "protein_clustering",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h", "aho_corasick/*.h"]),
    copts = tf_copts() + ["-std=c++14"],
    #linkopts = ["-lrados"],
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [":swps3_lib",
            "//tensorflow/core:framework_headers_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            "//tensorflow/contrib/persona/kernels/concurrent_queue:concurrent_queue"
            ]
)
