# user ops

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agd_remote_reader",
    srcs = glob(["agd_remote_record_reader.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkopts = [
        "-lz",
        "-lrados",
        "-lboost_system",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/contrib/persona/kernels/snap-align:snap_lib",
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        #"//tensorflow/contrib/persona/kernels/lttng:trace",
        "//tensorflow/contrib/persona/kernels/agd-format/proto:alignment_proto_cc",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

# copy this
cc_library(
    name = "agd_format",
    srcs = glob(["*.cc"], exclude=["agd_remote_record_reader.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkopts = [
        "-lz",
        "-lboost_system","-L/usr/local/lib"
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/contrib/persona/kernels/snap-align:snap_lib",
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        #"//tensorflow/contrib/persona/kernels/lttng:trace",
        "//tensorflow/contrib/persona/kernels/agd-format/proto:alignment_proto_cc",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf",
        ":agd_remote_reader",
    ],
    alwayslink = 1,
)
