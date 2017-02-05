# user ops

load("//tensorflow:tensorflow.bzl", "tf_copts")
# copy this
cc_library(
    name = "agd_format",
    srcs = glob(["*.cc"],
                exclude = ["parallel_sam_writer_op.cc", "zmq_source.cc"]),
    hdrs = glob(["*.h", "json/src/json.hpp"]),
    copts = tf_copts() + ["-fexceptions"],
    nocopts = "-fno-exceptions",
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework_headers_lib",
            #"//tensorflow/core/kernels:io",
            "//tensorflow/contrib/persona/kernels/snap-align:snap_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/lttng:trace",
            "//third_party/eigen3",
            "@protobuf//:protobuf",
            #"//tensorflow/core/kernels:ops_util"
            ],
    # TODO: this needs to be built better, like this: https://github.com/mzhaom/trunk/blob/master/third_party/boost/BUILD
    linkopts = [
        "-lz", "-ls3", "-lrados", "-lboost_system"
        ]
    )

cc_library(
    name = "zmq_source",
    srcs = ["zmq_source.cc"],
    copts = tf_copts() + ["-fexceptions"],
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework_headers_lib", 
            "//tensorflow/contrib/persona/kernels/lttng:trace",
            "//third_party/eigen3",
            "@protobuf//:protobuf",
    ],
    linkopts = [
           "-lzmq"
    ]
)

#cc_library(
#    name = "sam_writer",
#    srcs = ["parallel_sam_writer_op.cc"],
#    #hdrs = ["read_resource.h"],
#    includes=["."],
#    copts = tf_copts() + ["-fexceptions"],
#    nocopts = "-fno-exceptions",
#    linkstatic = 1,
#    alwayslink = 1,
#    visibility = ["//visibility:public"],
#    deps = [
#    "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
#    "//tensorflow/contrib/persona/kernels/dna-align:snap_align",
#    ],
#    )
