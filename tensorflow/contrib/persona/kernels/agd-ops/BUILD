load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agd_ceph_ops",
    srcs = glob(["agd_ceph_*.cc", "ceph_*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    linkopts = ["-lrados"],
    #linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format"],
)

cc_library(
    name = "agd_ops",
    srcs = glob(["*.cc"], exclude=["agd_ceph_*.cc", "ceph*.cc"]),
    hdrs = glob(["*.h"], exclude=["agd_ceph_reader.h, agd_ceph_writer.h"]),
    copts = tf_copts() + ["-fexceptions", "-I/usr/local/include", "-L/usr/local/include"],
    #includes = ["/usr/local/include/boost"],
    #linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    linkopts = [
        "-lboost_system","-L/usr/local/lib"
    ],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format"],
)
