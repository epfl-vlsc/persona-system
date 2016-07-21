# Tracepoints

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
  name = "trace",
  srcs = glob(["*.cc"]),
  hdrs = glob(["*.h"]),
  copts = tf_copts(),
  linkstatic = 1,
  alwayslink = 1,
  visibility = ["//visibility:public"],
  linkopts = [
           "-llttng-ust", "-ldl"
  ]
)