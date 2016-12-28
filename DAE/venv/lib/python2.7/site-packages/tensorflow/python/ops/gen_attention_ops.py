"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def extract_glimpse(input, size, offsets, centered=None, normalized=None,
                    uniform_noise=None, name=None):
  r"""Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location `offsets`
  from the input tensor. If the windows only partially overlaps the inputs, the
  non overlapping areas will be filled with random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the same as that
  of the input tensor. The height and width of the output windows are
  specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:
  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width dimension.
  * If the coordinates are both normalized and centered, they range from -1.0 to
    1.0. The coordinates (-1.0, -1.0) correspond to the upper left corner, the
    lower right corner is located at  (1.0, 1.0) and the center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements containing the size of the glimpses to extract.
      The glimpse height must be specified first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`.
      A 2-D integer tensor of shape `[batch_size, 2]` containing the x, y
      locations of the center of each window.
    centered: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are centered relative to
      the image, in which case the (0, 0) offset is relative to the center of the
      input images. If false, the (0,0) offset corresponds to the upper left corner
      of the input images.
    normalized: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`.
      indicates if the noise should be generated using a
      uniform distribution or a gaussian distribution.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A tensor representing the glimpses `[batch_size, glimpse_height,
    glimpse_width, channels]`.
  """
  return _op_def_lib.apply_op("ExtractGlimpse", input=input, size=size,
                              offsets=offsets, centered=centered,
                              normalized=normalized,
                              uniform_noise=uniform_noise, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "ExtractGlimpse"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  input_arg {
    name: "offsets"
    type: DT_FLOAT
  }
  output_arg {
    name: "glimpse"
    type: DT_FLOAT
  }
  attr {
    name: "centered"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "normalized"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "uniform_noise"
    type: "bool"
    default_value {
      b: true
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
