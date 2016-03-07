cc_binary(
    name = "dna_align.so",
    includes = [".", "snap/SNAPLib"],
    srcs = glob(["*.cc"]),
	#copts = ["-w"],
    linkopts = [
        "-Wl,-Bsymbolic",
        "-lm",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
)