"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def adjust_contrast(images, contrast_factor, min_value, max_value, name=None):
  r"""Adjust the contrast of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, the Op first computes the mean of the image pixels in the
  channel and then adjusts each component of each pixel to
  `(x - mean) * contrast_factor + mean`.

  These adjusted values are then clipped to fit in the `[min_value, max_value]`
  interval.

  `images: Images to adjust.  At least 3-D.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
    contrast_factor: A `Tensor` of type `float32`.
      A float multiplier for adjusting contrast.
    min_value: A `Tensor` of type `float32`.
      Minimum value for clipping the adjusted pixels.
    max_value: A `Tensor` of type `float32`.
      Maximum value for clipping the adjusted pixels.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. The constrast-adjusted image or images.
  """
  return _op_def_lib.apply_op("AdjustContrast", images=images,
                              contrast_factor=contrast_factor,
                              min_value=min_value, max_value=max_value,
                              name=name)


def decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None,
                try_recover_truncated=None, acceptable_fraction=None,
                name=None):
  r"""Decode a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`..
  """
  return _op_def_lib.apply_op("DecodeJpeg", contents=contents,
                              channels=channels, ratio=ratio,
                              fancy_upscaling=fancy_upscaling,
                              try_recover_truncated=try_recover_truncated,
                              acceptable_fraction=acceptable_fraction,
                              name=name)


def decode_png(contents, channels=None, name=None):
  r"""Decode a PNG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the PNG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  If needed, the PNG-encoded image is transformed to match the requested number
  of color channels.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`.
  """
  return _op_def_lib.apply_op("DecodePng", contents=contents,
                              channels=channels, name=name)


def encode_jpeg(image, format=None, quality=None, progressive=None,
                optimize_size=None, chroma_downsampling=None,
                density_unit=None, x_density=None, y_density=None,
                xmp_metadata=None, name=None):
  r"""JPEG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

  The attr `format` can be used to override the color format of the encoded
  output.  Values can be:

  *   `''`: Use a default format based on the number of channels in the image.
  *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
      of `image` must be 1.
  *   `rgb`: Output an RGB JPEG image. The `channels` dimension
      of `image` must be 3.

  If `format` is not specified or is the empty string, a default format is picked
  in function of the number of channels in `image`:

  *   1: Output a grayscale image.
  *   3: Output an RGB image.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    format: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`.
      Per pixel image format.
    quality: An optional `int`. Defaults to `95`.
      Quality of the compression from 0 to 100 (higher is better and slower).
    progressive: An optional `bool`. Defaults to `False`.
      If True, create a JPEG that loads progressively (coarse to fine).
    optimize_size: An optional `bool`. Defaults to `False`.
      If True, spend CPU/RAM to reduce size with no quality change.
    chroma_downsampling: An optional `bool`. Defaults to `True`.
      See http://en.wikipedia.org/wiki/Chroma_subsampling.
    density_unit: An optional `string` from: `"in", "cm"`. Defaults to `"in"`.
      Unit used to specify `x_density` and `y_density`:
      pixels per inch (`'in'`) or centimeter (`'cm'`).
    x_density: An optional `int`. Defaults to `300`.
      Horizontal pixels per density unit.
    y_density: An optional `int`. Defaults to `300`.
      Vertical pixels per density unit.
    xmp_metadata: An optional `string`. Defaults to `""`.
      If not empty, embed this XMP metadata in the image header.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. 0-D. JPEG-encoded image.
  """
  return _op_def_lib.apply_op("EncodeJpeg", image=image, format=format,
                              quality=quality, progressive=progressive,
                              optimize_size=optimize_size,
                              chroma_downsampling=chroma_downsampling,
                              density_unit=density_unit, x_density=x_density,
                              y_density=y_density, xmp_metadata=xmp_metadata,
                              name=name)


def encode_png(image, compression=None, name=None):
  r"""PNG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]` where
  `channels` is:

  *   1: for grayscale.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level, generating
  the smallest output, but is slower.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. 0-D. PNG-encoded image.
  """
  return _op_def_lib.apply_op("EncodePng", image=image,
                              compression=compression, name=name)


def random_crop(image, size, seed=None, seed2=None, name=None):
  r"""Randomly crop `image`.

  `size` is a 1-D int64 tensor with 2 elements representing the crop height and
  width.  The values must be non negative.

  This Op picks a random location in `image` and crops a `height` by `width`
  rectangle from that location.  The random location is picked so the cropped
  area will fit inside the original image.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      3-D of shape `[height, width, channels]`.
    size: A `Tensor` of type `int64`.
      1-D of length 2 containing: `crop_height`, `crop_width`..
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `image`.
    3-D of shape `[crop_height, crop_width, channels].`
  """
  return _op_def_lib.apply_op("RandomCrop", image=image, size=size, seed=seed,
                              seed2=seed2, name=name)


def resize_area(images, size, name=None):
  r"""Resize `images` to `size` using area interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeArea", images=images, size=size,
                              name=name)


def resize_bicubic(images, size, name=None):
  r"""Resize `images` to `size` using bicubic interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeBicubic", images=images, size=size,
                              name=name)


def resize_bilinear(images, size, name=None):
  r"""Resize `images` to `size` using bilinear interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeBilinear", images=images, size=size,
                              name=name)


def resize_nearest_neighbor(images, size, name=None):
  r"""Resize `images` to `size` using nearest neighbor interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeNearestNeighbor", images=images,
                              size=size, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AdjustContrast"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "contrast_factor"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_value"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_value"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "DecodeJpeg"
  input_arg {
    name: "contents"
    type: DT_STRING
  }
  output_arg {
    name: "image"
    type: DT_UINT8
  }
  attr {
    name: "channels"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "ratio"
    type: "int"
    default_value {
      i: 1
    }
  }
  attr {
    name: "fancy_upscaling"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "try_recover_truncated"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "acceptable_fraction"
    type: "float"
    default_value {
      f: 1
    }
  }
}
op {
  name: "DecodePng"
  input_arg {
    name: "contents"
    type: DT_STRING
  }
  output_arg {
    name: "image"
    type: DT_UINT8
  }
  attr {
    name: "channels"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "EncodeJpeg"
  input_arg {
    name: "image"
    type: DT_UINT8
  }
  output_arg {
    name: "contents"
    type: DT_STRING
  }
  attr {
    name: "format"
    type: "string"
    default_value {
      s: ""
    }
    allowed_values {
      list {
        s: ""
        s: "grayscale"
        s: "rgb"
      }
    }
  }
  attr {
    name: "quality"
    type: "int"
    default_value {
      i: 95
    }
  }
  attr {
    name: "progressive"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "optimize_size"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "chroma_downsampling"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "density_unit"
    type: "string"
    default_value {
      s: "in"
    }
    allowed_values {
      list {
        s: "in"
        s: "cm"
      }
    }
  }
  attr {
    name: "x_density"
    type: "int"
    default_value {
      i: 300
    }
  }
  attr {
    name: "y_density"
    type: "int"
    default_value {
      i: 300
    }
  }
  attr {
    name: "xmp_metadata"
    type: "string"
    default_value {
      s: ""
    }
  }
}
op {
  name: "EncodePng"
  input_arg {
    name: "image"
    type: DT_UINT8
  }
  output_arg {
    name: "contents"
    type: DT_STRING
  }
  attr {
    name: "compression"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "RandomCrop"
  input_arg {
    name: "image"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  is_stateful: true
}
op {
  name: "ResizeArea"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "ResizeBicubic"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "ResizeBilinear"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "ResizeNearestNeighbor"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
