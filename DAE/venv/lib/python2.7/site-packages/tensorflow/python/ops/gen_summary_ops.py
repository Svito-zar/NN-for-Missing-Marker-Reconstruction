"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _histogram_summary(tag, values, name=None):
  r"""Outputs a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `OutOfRange` error if any value is not finite.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar.  Tag to use for the `Summary.Value`.
    values: A `Tensor` of type `float32`.
      Any shape. Values to use to build the histogram.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  return _op_def_lib.apply_op("HistogramSummary", tag=tag, values=values,
                              name=name)


def _image_summary(tag, tensor, max_images=None, bad_color=None, name=None):
  r"""Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. Their values
  are normalized, one image at a time, to fit in the range `[0, 255]`.  The
  op uses two different normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  The `bad_color` argument is the color to use in the generated images for
  non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
  Each element must be in the range `[0, 255]` (It represents the value of a
  pixel in the output image).  Non-finite values in the input tensor are
  replaced by this tensor in the output image.  The default value is the color
  red.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`.
      4-D of shape `[batch_size, height, width, channels]` where
      `channels` is 1, 3, or 4.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate images for.
    bad_color: . Defaults to `[]`.
      Color to use for pixels with non-finite values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  return _op_def_lib.apply_op("ImageSummary", tag=tag, tensor=tensor,
                              max_images=max_images, bad_color=bad_color,
                              name=name)


def _merge_summary(inputs, name=None):
  r"""Merges summaries.

  This op creates a
  [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of at least 1 `Tensor` objects of type `string`.
      Can be of any shape.  Each must contain serialized `Summary` protocol
      buffers.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  return _op_def_lib.apply_op("MergeSummary", inputs=inputs, name=name)


def _scalar_summary(tags, values, name=None):
  r"""Outputs a `Summary` protocol buffer with scalar values.

  The input `tags` and `values` must have the same shape.  The generated summary
  has a summary value for each tag-value pair in `tags` and `values`.

  Args:
    tags: A `Tensor` of type `string`. 1-D. Tags for the summary.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      1-D, same size as `tags.  Values for the summary.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    Scalar.  Serialized `Summary` protocol buffer.
  """
  return _op_def_lib.apply_op("ScalarSummary", tags=tags, values=values,
                              name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "HistogramSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type: DT_FLOAT
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
}
op {
  name: "ImageSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type: DT_FLOAT
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "max_images"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "bad_color"
    type: "tensor"
    default_value {
      tensor {
        dtype: DT_UINT8
        tensor_shape {
          dim {
            size: 4
          }
        }
        int_val: 255
        int_val: 0
        int_val: 0
        int_val: 255
      }
    }
  }
}
op {
  name: "MergeSummary"
  input_arg {
    name: "inputs"
    type: DT_STRING
    number_attr: "N"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "ScalarSummary"
  input_arg {
    name: "tags"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
