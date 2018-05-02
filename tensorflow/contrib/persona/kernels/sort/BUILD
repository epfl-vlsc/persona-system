# sorting operations

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agd_ceph_merge",
    srcs = ["agd_ceph_merge_op.cc"],
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkopts = ["-lrados"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format"],
)

cc_library(
    name = "agd_sort",
    srcs = glob(["*.cc"], exclude=["agd_ceph_merge_op.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format"],
)
