load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "bamtools",
    srcs = glob(["freebayes/bamtools/lib/libbamtools.a"]),
    includes = ["freebayes/bamtools/src/api/" , "freebayes/bamtools/src/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)

cc_library(
    name = "ttmath",
    srcs = glob(["freebayes/ttmath/*.cpp"]),
    hdrs = glob(["freebayes/ttmath/*.h"]),
    includes = ["freebayes/ttmath/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)
cc_library(
    name = "intervaltree",
    srcs = glob(["freebayes/intervaltree/*.cpp"]),
    hdrs = glob(["freebayes/intervaltree/*.h"]),
    includes = ["freebayes/intervaltree/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)

cc_library(
    name = "seqlib",
    srcs = glob(["freebayes/SeqLib/src/libseqlib.a"]),
    includes = ["freebayes/SeqLib/SeqLib/","freebayes/SeqLib/" , "freebayes/SeqLib/src/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [":htslib" , ":bamtools"]
)

cc_library(
    name = "fermi_lite",
    srcs = glob(["freebayes/SeqLib/fermi-lite/*.c"] , exclude = ["freebayes/SeqLib/fermi-lite/example.c"]),
    hdrs = glob(["freebayes/SeqLib/fermilite/*.h"]),
    includes = ["freebayes/SeqLib/fermi-lite/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/bwa-align:bwa_align"]
)


cc_library(
    name = "htslib",
    srcs = glob(["freebayes/SeqLib/htslib/*.a"]),
    #srcs = glob(["freebayes/vcflib/tabixpp/htslib/htslib/*.cpp","freebayes/vcflib/tabixpp/htslib/*.c" ,"freebayes/vcflib/tabixpp/htslib/*.cpp" ]),
    #hdrs = glob(["freebayes/vcflib/tabixpp/htslib/htslib/*.h" , "freebayes/vcflib/tabixpp/htslib/*.h"]),
    includes = ["freebayes/vcflib/tabixpp/htslib/htslib/" ,"freebayes/vcflib/tabixpp/htslib/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)


cc_library(
    name = "tabixpp",
    srcs = glob(["freebayes/vcflib/tabixpp/*.cpp","freebayes/vcflib/tabixpp/*.hpp" ]),
    hdrs = glob(["freebayes/vcflib/tabixpp/*.h","freebayes/vcflib/tabixpp/*.hpp"]),
    includes = ["freebayes/vcflib/tabixpp/","freebayes/vcflib/tabixpp/htslib/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [":htslib"]
)

cc_library(
    name = "smithwaterman",
    srcs = glob(["freebayes/vcflib/smithwaterman/*.cpp" ]),
    hdrs = glob(["freebayes/vcflib/smithwaterman/*.h" ]),
    includes = ["freebayes/vcflib/smithwaterman/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "multichoose",
    srcs = glob(["freebayes/vcflib/multichoose/*.cpp" ]),
    hdrs = glob(["freebayes/vcflib/multichoose/*.h" ]),
    includes = ["freebayes/vcflib/multichoose/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)
cc_library(
    name = "fastahack",
    srcs = glob(["freebayes/vcflib/fastahack/*.cpp" ]),
    hdrs = glob(["freebayes/vcflib/fastahack/*.h" ]),
    includes = ["freebayes/vcflib/fastahack/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"]
)

cc_library(
    name = "fsom",
    srcs = glob(["freebayes/vcflib/fsom/*.cpp" ]),
    hdrs = glob(["freebayes/vcflib/fsom/*.h" ]),
    includes = ["freebayes/vcflib/fsom/" ] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [":fastahack"]
)
cc_library(
    name = "vcflib",
    srcs = glob(["freebayes/vcflib/src/*.cpp",  "freebayes/vcflib/filevercmp/*.cpp" ]),
    hdrs = glob(["freebayes/vcflib/src/*.h", "freebayes/vcflib/filevercmp/*.h" ]),
    includes = ["freebayes/vcflib/src/",  "freebayes/vcflib/filevercmp/" , "freebayes/vcflib/"] ,
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [":tabixpp",":smithwaterman", ":fastahack",":multichoose", ":intervaltree" , ":fsom"]
)

cc_library(
    name = "AGDMultiReader",
    srcs = glob(["AGDMultiReader.cc"]),
    hdrs = glob(["AGDMultiReader.h"]),
    copts = tf_copts() + ["-fexceptions"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format", ":bamtools",":seqlib"],
    alwayslink=1,
    linkopts = [
             "-lboost_system", "-lboost_timer" , "-lrados"
    ]
)

# Compilation fo the freebayes library
cc_library(
    name = "freebayes",
    #srcs = glob(["freebayes.a"]),
    srcs = glob(["freebayes/src/*.cpp"]) ,
    hdrs = glob(["freebayes/src/*.h" ]),
    includes = ["freebayes/src/"] ,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"],
    deps = [":AGDMultiReader",":bamtools" , ":intervaltree" , ":seqlib" , ":ttmath", ":vcflib"  , "//tensorflow/contrib/persona/kernels/bwa-align:bwa_align", ":fermi_lite"]

)



cc_library(
    name = "variant_calling",
    srcs = glob(["variant_calling.cc"]),
    #hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/agd-format:agd_format", ":freebayes" , ":AGDMultiReader"],
    alwayslink=1,
    linkopts = [
             "-lboost_system", "-lboost_timer" , "-lrados"
    ]
)
