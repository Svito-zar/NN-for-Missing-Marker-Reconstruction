"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _abs(x, name=None):
  r"""Computes the absolute value of a tensor.

  Given a tensor `x`, this operation returns a tensor containing the absolute
  value of each element in `x`. For example, if x is an input element and y is
  an output element, this operation computes \\(y = |x|\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Abs", x=x, name=name)


def add(x, y, name=None):
  r"""Returns x + y element-wise.

  *NOTE*: Add supports broadcasting. AddN does not.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int8`, `int16`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Add", x=x, y=y, name=name)


def add_n(inputs, name=None):
  r"""Add all input tensors element wise.

  Args:
    inputs: A list of at least 1 `Tensor` objects of the same type in: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Must all be the same size and shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  return _op_def_lib.apply_op("AddN", inputs=inputs, name=name)


def _all(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The reduced tensor.
  """
  return _op_def_lib.apply_op("All", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def _any(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Any", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def arg_max(input, dimension, name=None):
  r"""Returns the index with the largest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  return _op_def_lib.apply_op("ArgMax", input=input, dimension=dimension,
                              name=name)


def arg_min(input, dimension, name=None):
  r"""Returns the index with the smallest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  return _op_def_lib.apply_op("ArgMin", input=input, dimension=dimension,
                              name=name)


def _batch_mat_mul(x, y, adj_x=None, adj_y=None, name=None):
  r"""Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`.
      3-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      3-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
    3-D or higher with shape `[..., r_o, c_o]`
  """
  return _op_def_lib.apply_op("BatchMatMul", x=x, y=y, adj_x=adj_x,
                              adj_y=adj_y, name=name)


def cast(x, DstT, name=None):
  r"""Cast x of type SrcT to y of DstT.

  Args:
    x: A `Tensor`.
    DstT: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `DstT`.
  """
  return _op_def_lib.apply_op("Cast", x=x, DstT=DstT, name=name)


def ceil(x, name=None):
  r"""Returns element-wise smallest integer in not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Ceil", x=x, name=name)


def _complex(real, imag, name=None):
  r"""Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor` of type `float32`.
    imag: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  return _op_def_lib.apply_op("Complex", real=real, imag=imag, name=name)


def complex_abs(x, name=None):
  r"""Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` that is the absolute value of each element in `x`. All elements in `x`
  must be complex numbers of the form \\(a + bj\\). The absolute value is
  computed as \\( \sqrt{a^2 + b^2}\\).

  For example:

  ```
  # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
  tf.complex_abs(x) ==> [5.25594902, 6.60492229]
  ```

  Args:
    x: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return _op_def_lib.apply_op("ComplexAbs", x=x, name=name)


def conj(in_, name=None):
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `in`. The
  complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
  part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  return _op_def_lib.apply_op("Conj", in_=in_, name=name)


def cos(x, name=None):
  r"""Computes cos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Cos", x=x, name=name)


def div(x, y, name=None):
  r"""Returns x / y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Div", x=x, y=y, name=name)


def equal(x, y, name=None):
  r"""Returns the truth value of (x == y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("Equal", x=x, y=y, name=name)


def exp(x, name=None):
  r"""Computes exponential of x element-wise.  \\(y = e^x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Exp", x=x, name=name)


def floor(x, name=None):
  r"""Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Floor", x=x, name=name)


def greater(x, y, name=None):
  r"""Returns the truth value of (x > y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("Greater", x=x, y=y, name=name)


def greater_equal(x, y, name=None):
  r"""Returns the truth value of (x >= y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("GreaterEqual", x=x, y=y, name=name)


def imag(in_, name=None):
  r"""Returns the imaginary part of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of type
  `float` that is the imaginary part of each element in `in`. All elements in `in`
  must be complex numbers of the form \\(a + bj\\), where *a* is the real part
  and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(in) ==> [4.75, 5.75]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return _op_def_lib.apply_op("Imag", in_=in_, name=name)


def inv(x, name=None):
  r"""Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Inv", x=x, name=name)


def is_finite(x, name=None):
  r"""Returns which elements of x are finite.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("IsFinite", x=x, name=name)


def is_inf(x, name=None):
  r"""Returns which elements of x are Inf.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("IsInf", x=x, name=name)


def is_nan(x, name=None):
  r"""Returns which elements of x are NaN.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("IsNan", x=x, name=name)


def less(x, y, name=None):
  r"""Returns the truth value of (x < y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("Less", x=x, y=y, name=name)


def less_equal(x, y, name=None):
  r"""Returns the truth value of (x <= y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("LessEqual", x=x, y=y, name=name)


def lin_space(start, stop, num, name=None):
  r"""Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      Last entry in the range.
    num: A `Tensor` of type `int32`. Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`. 1-D. The generated values.
  """
  return _op_def_lib.apply_op("LinSpace", start=start, stop=stop, num=num,
                              name=name)


def log(x, name=None):
  r"""Computes natural logrithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Log", x=x, name=name)


def logical_and(x, y, name=None):
  r"""Returns the truth value of x AND y element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("LogicalAnd", x=x, y=y, name=name)


def logical_not(x, name=None):
  r"""Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("LogicalNot", x=x, name=name)


def logical_or(x, y, name=None):
  r"""Returns the truth value of x OR y element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("LogicalOr", x=x, y=y, name=name)


def _mat_mul(a, b, transpose_a=None, transpose_b=None, name=None):
  r"""Multiply the matrix "a" by the matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of
  "a" (after being transposed if transpose_a is true) must match the
  outer dimension of "b" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, "a" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, "b" is transposed before multiplication.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  return _op_def_lib.apply_op("MatMul", a=a, b=b, transpose_a=transpose_a,
                              transpose_b=transpose_b, name=name)


def _max(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the maximum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Max", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def maximum(x, y, name=None):
  r"""Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Maximum", x=x, y=y, name=name)


def _mean(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the mean of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Mean", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def _min(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the minimum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Min", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def minimum(x, y, name=None):
  r"""Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Minimum", x=x, y=y, name=name)


def mod(x, y, name=None):
  r"""Returns element-wise remainder of division.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Mod", x=x, y=y, name=name)


def mul(x, y, name=None):
  r"""Returns x * y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int8`, `int16`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Mul", x=x, y=y, name=name)


def neg(x, name=None):
  r"""Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Neg", x=x, name=name)


def not_equal(x, y, name=None):
  r"""Returns the truth value of (x != y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  return _op_def_lib.apply_op("NotEqual", x=x, y=y, name=name)


def _pow(x, y, name=None):
  r"""Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Pow", x=x, y=y, name=name)


def _prod(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the product of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Prod", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def _range(start, limit, delta, name=None):
  r"""Creates a sequence of integers.

  This operation creates a sequence of integers that begins at `start` and
  extends by increments of `delta` up to but not including `limit`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  ```

  Args:
    start: A `Tensor` of type `int32`.
      0-D (scalar). First entry in the sequence.
    limit: A `Tensor` of type `int32`.
      0-D (scalar). Upper limit of sequence, exclusive.
    delta: A `Tensor` of type `int32`.
      0-D (scalar). Optional. Default is 1. Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. 1-D.
  """
  return _op_def_lib.apply_op("Range", start=start, limit=limit, delta=delta,
                              name=name)


def real(in_, name=None):
  r"""Returns the real part of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of type
  `float` that is the real part of each element in `in`. All elements in `in`
  must be complex numbers of the form \\(a + bj\\), where *a* is the real part
  returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(in) ==> [-2.25, 3.25]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return _op_def_lib.apply_op("Real", in_=in_, name=name)


def rsqrt(x, name=None):
  r"""Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Rsqrt", x=x, name=name)


def segment_max(data, segment_ids, name=None):
  r"""Computes the maximum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/SegmentMax.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SegmentMax", data=data,
                              segment_ids=segment_ids, name=name)


def segment_mean(data, segment_ids, name=None):
  r"""Computes the mean along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/SegmentMean.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SegmentMean", data=data,
                              segment_ids=segment_ids, name=name)


def segment_min(data, segment_ids, name=None):
  r"""Computes the minimum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/SegmentMin.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SegmentMin", data=data,
                              segment_ids=segment_ids, name=name)


def segment_prod(data, segment_ids, name=None):
  r"""Computes the product along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/SegmentProd.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SegmentProd", data=data,
                              segment_ids=segment_ids, name=name)


def segment_sum(data, segment_ids, name=None):
  r"""Computes the sum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/SegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SegmentSum", data=data,
                              segment_ids=segment_ids, name=name)


def select(condition, t, e, name=None):
  r"""Selects elements from `t` or `e`, depending on `condition`.

  The `condition`, `t`, and `e` tensors must all have the same shape,
  and the output will also have that shape. The `condition` tensor acts
  as an element-wise mask that chooses, based on the value at each
  element, whether the corresponding element in the output should be
  taken from `t` (if true) or `e` (if false). For example:

  For example:

  ```prettyprint
  # 'condition' tensor is [[True, False]
  #                        [True, False]]
  # 't' is [[1, 1],
  #         [1, 1]]
  # 'e' is [[2, 2],
  #         [2, 2]]
  select(condition, t, e) ==> [[1, 2],
                               [1, 2]]
  ```

  Args:
    condition: A `Tensor` of type `bool`.
    t:  A `Tensor` with the same shape as `condition`.
    e:  A `Tensor` with the same type and shape as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type and shape as `t` and `e`.
  """
  return _op_def_lib.apply_op("Select", condition=condition, t=t, e=e,
                              name=name)


def _sigmoid(x, name=None):
  r"""Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Sigmoid", x=x, name=name)


def sign(x, name=None):
  r"""Returns an element-wise indication of the sign of a number.

  y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Sign", x=x, name=name)


def sin(x, name=None):
  r"""Computes sin of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Sin", x=x, name=name)


def _sparse_mat_mul(a, b, transpose_a=None, transpose_b=None,
                    a_is_sparse=None, b_is_sparse=None, name=None):
  r"""Multiply matrix "a" by matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of "a" must
  match the outer dimension of "b". This op is optimized for the case where at
  least one of "a" or "b" is sparse. The breakeven for using this versus a dense
  matrix multiply on one platform was 30% zero values in the sparse matrix.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return _op_def_lib.apply_op("SparseMatMul", a=a, b=b,
                              transpose_a=transpose_a,
                              transpose_b=transpose_b,
                              a_is_sparse=a_is_sparse,
                              b_is_sparse=b_is_sparse, name=name)


def sparse_segment_mean(data, indices, segment_ids, name=None):
  r"""Computes the mean along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension_0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor` of type `int32`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SparseSegmentMean", data=data, indices=indices,
                              segment_ids=segment_ids, name=name)


def sparse_segment_mean_grad(grad, indices, segment_ids, output_dim0,
                             name=None):
  r"""Computes gradients for SparseSegmentMean.

  Returns tensor "output" with same shape as grad, except for dimension_0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor` of type `int32`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension_0 of "data" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  return _op_def_lib.apply_op("SparseSegmentMeanGrad", grad=grad,
                              indices=indices, segment_ids=segment_ids,
                              output_dim0=output_dim0, name=name)


def sparse_segment_sum(data, indices, segment_ids, name=None):
  r"""Computes the sum along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension_0, specified by `indices`.

  For example:

  ```prettyprint
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
    ==> [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
    ==> [[ 1  2  3  4]
         [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
    ==> [[0 0 0 0]
         [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    indices: A `Tensor` of type `int32`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `k`, the number of segments.
  """
  return _op_def_lib.apply_op("SparseSegmentSum", data=data, indices=indices,
                              segment_ids=segment_ids, name=name)


def sqrt(x, name=None):
  r"""Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Sqrt", x=x, name=name)


def square(x, name=None):
  r"""Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Square", x=x, name=name)


def sub(x, y, name=None):
  r"""Returns x - y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Sub", x=x, y=y, name=name)


def _sum(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the sum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      The tensor to reduce.
    reduction_indices: A `Tensor` of type `int32`. The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  return _op_def_lib.apply_op("Sum", input=input,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims, name=name)


def _tanh(x, name=None):
  r"""Computes hyperbolic tangent of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Tanh", x=x, name=name)


def unsorted_segment_sum(data, segment_ids, num_segments, name=None):
  r"""Computes the sum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
    range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  `num_segments` should equal the number of distinct segment IDs.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/UnsortedSegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.
    num_segments: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension_0 which
    has size `num_segments`.
  """
  return _op_def_lib.apply_op("UnsortedSegmentSum", data=data,
                              segment_ids=segment_ids,
                              num_segments=num_segments, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Abs"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Add"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "AddN"
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "sum"
    type_attr: "T"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
  is_aggregate: true
  is_commutative: true
}
op {
  name: "All"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "Any"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ArgMax"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dimension"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "ArgMin"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dimension"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "BatchMatMul"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
      }
    }
  }
  attr {
    name: "adj_x"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "adj_y"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "Cast"
  input_arg {
    name: "x"
    type_attr: "SrcT"
  }
  output_arg {
    name: "y"
    type_attr: "DstT"
  }
  attr {
    name: "SrcT"
    type: "type"
  }
  attr {
    name: "DstT"
    type: "type"
  }
}
op {
  name: "Ceil"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
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
op {
  name: "Complex"
  input_arg {
    name: "real"
    type: DT_FLOAT
  }
  input_arg {
    name: "imag"
    type: DT_FLOAT
  }
  output_arg {
    name: "out"
    type: DT_COMPLEX64
  }
}
op {
  name: "ComplexAbs"
  input_arg {
    name: "x"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "y"
    type: DT_FLOAT
  }
}
op {
  name: "Conj"
  input_arg {
    name: "in"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "out"
    type: DT_COMPLEX64
  }
}
op {
  name: "Cos"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Div"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Equal"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_QUINT8
        type: DT_QINT8
        type: DT_QINT32
      }
    }
  }
  is_commutative: true
}
op {
  name: "Exp"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Floor"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
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
op {
  name: "Greater"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "GreaterEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Imag"
  input_arg {
    name: "in"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "out"
    type: DT_FLOAT
  }
}
op {
  name: "Inv"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "IsFinite"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
op {
  name: "IsInf"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
op {
  name: "IsNan"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
op {
  name: "Less"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "LessEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "LinSpace"
  input_arg {
    name: "start"
    type_attr: "T"
  }
  input_arg {
    name: "stop"
    type_attr: "T"
  }
  input_arg {
    name: "num"
    type: DT_INT32
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
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Log"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "LogicalAnd"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  input_arg {
    name: "y"
    type: DT_BOOL
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  is_commutative: true
}
op {
  name: "LogicalNot"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  output_arg {
    name: "y"
    type: DT_BOOL
  }
}
op {
  name: "LogicalOr"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  input_arg {
    name: "y"
    type: DT_BOOL
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  is_commutative: true
}
op {
  name: "MatMul"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  output_arg {
    name: "product"
    type_attr: "T"
  }
  attr {
    name: "transpose_a"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "transpose_b"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
      }
    }
  }
}
op {
  name: "Max"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Maximum"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "Mean"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Min"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Minimum"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "Mod"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Mul"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "Neg"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "NotEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_QUINT8
        type: DT_QINT8
        type: DT_QINT32
      }
    }
  }
  is_commutative: true
}
op {
  name: "Pow"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Prod"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Range"
  input_arg {
    name: "start"
    type: DT_INT32
  }
  input_arg {
    name: "limit"
    type: DT_INT32
  }
  input_arg {
    name: "delta"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
}
op {
  name: "Real"
  input_arg {
    name: "in"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "out"
    type: DT_FLOAT
  }
}
op {
  name: "Rsqrt"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SegmentMax"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SegmentMean"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SegmentMin"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SegmentProd"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Select"
  input_arg {
    name: "condition"
    type: DT_BOOL
  }
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "e"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Sigmoid"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Sign"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Sin"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "SparseMatMul"
  input_arg {
    name: "a"
    type: DT_FLOAT
  }
  input_arg {
    name: "b"
    type: DT_FLOAT
  }
  output_arg {
    name: "product"
    type: DT_FLOAT
  }
  attr {
    name: "transpose_a"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "transpose_b"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "a_is_sparse"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "b_is_sparse"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseSegmentMean"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
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
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SparseSegmentMeanGrad"
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
  }
  input_arg {
    name: "output_dim0"
    type: DT_INT32
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
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SparseSegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
}
op {
  name: "Sqrt"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Square"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Sub"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Sum"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Tanh"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_INT64
      }
    }
  }
}
op {
  name: "UnsortedSegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
  }
  input_arg {
    name: "num_segments"
    type: DT_INT32
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
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
