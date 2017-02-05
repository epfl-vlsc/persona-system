
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_persona_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_persona_ops.so"))

def persona_ops():
    return _persona_ops

