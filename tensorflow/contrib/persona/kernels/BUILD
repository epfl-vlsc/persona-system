# Description:
#   Contains kernels for Persona.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "persona_kernels",
    deps = [
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
        "//tensorflow/contrib/persona/kernels/agd-format:zmq_source",
        "//tensorflow/contrib/persona/kernels/snap-align:snap_align",
        "//tensorflow/contrib/persona/kernels/bwa-align:bwa_align",
        "//tensorflow/contrib/persona/kernels/bam:agdbam",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
