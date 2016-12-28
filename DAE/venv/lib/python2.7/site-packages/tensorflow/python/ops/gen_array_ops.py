"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _broadcast_gradient_args(s0, s1, name=None):
  r"""Return the reduction indices for computing gradients of s0 op s1 with broadcast.

  This is typically used by gradient computations for a broadcasting operation.

  Args:
    s0: A `Tensor` of type `int32`.
    s1: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r0, r1).
    r0: A `Tensor` of type `int32`.
    r1: A `Tensor` of type `int32`.
  """
  return _op_def_lib.apply_op("BroadcastGradientArgs", s0=s0, s1=s1,
                              name=name)


def check_numerics(tensor, message, name=None):
  r"""Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

  Args:
    tensor: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  return _op_def_lib.apply_op("CheckNumerics", tensor=tensor, message=message,
                              name=name)


def _concat(concat_dim, values, name=None):
  r"""Concatenates tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects of the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
    A `Tensor` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.
  """
  return _op_def_lib.apply_op("Concat", concat_dim=concat_dim, values=values,
                              name=name)


def _const(value, dtype, name=None):
  r"""Returns a constant tensor.

  Args:
    value: . Attr `value` is the tensor to return.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  return _op_def_lib.apply_op("Const", value=value, dtype=dtype, name=name)


def diag(diagonal, name=None):
  r"""Returns a diagonal tensor with a given diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
  rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

  `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

  For example:

  ```prettyprint
  # 'diagonal' is [1, 2, 3, 4]
  tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
  ```

  Args:
    diagonal: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Rank k tensor where k is at most 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  return _op_def_lib.apply_op("Diag", diagonal=diagonal, name=name)


def _edit_distance(hypothesis_indices, hypothesis_values, hypothesis_shape,
                   truth_indices, truth_values, truth_shape, normalize=None,
                   name=None):
  r"""Computes the (possibly normalized) Levenshtein Edit Distance.

  The inputs are variable-length sequences provided by SparseTensors
    (hypothesis_indices, hypothesis_values, hypothesis_shape)
  and
    (truth_indices, truth_values, truth_shape).

  The inputs are:

  Args:
    hypothesis_indices: A `Tensor` of type `int64`.
      The indices of the hypothesis list SparseTensor.
      This is an N x R int64 matrix.
    hypothesis_values: A `Tensor`.
      The values of the hypothesis list SparseTensor.
      This is an N-length vector.
    hypothesis_shape: A `Tensor` of type `int64`.
      The shape of the hypothesis list SparseTensor.
      This is an R-length vector.
    truth_indices: A `Tensor` of type `int64`.
      The indices of the truth list SparseTensor.
      This is an M x R int64 matrix.
    truth_values: A `Tensor`. Must have the same type as `hypothesis_values`.
      The values of the truth list SparseTensor.
      This is an M-length vector.
    truth_shape: A `Tensor` of type `int64`. truth indices, vector.
    normalize: An optional `bool`. Defaults to `True`.
      boolean (if true, edit distances are normalized by length of truth).

      The output is:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. A dense float tensor with rank R - 1.

    For the example input:

        // hypothesis represents a 2x1 matrix with variable-length values:
        //   (0,0) = ["a"]
        //   (1,0) = ["b"]
        hypothesis_indices = [[0, 0, 0],
                              [1, 0, 0]]
        hypothesis_values = ["a", "b"]
        hypothesis_shape = [2, 1, 1]

        // truth represents a 2x2 matrix with variable-length values:
        //   (0,0) = []
        //   (0,1) = ["a"]
        //   (1,0) = ["b", "c"]
        //   (1,1) = ["a"]
        truth_indices = [[0, 1, 0],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0]]
        truth_values = ["a", "b", "c", "a"]
        truth_shape = [2, 2, 2]
        normalize = true

    The output will be:

        // output is a 2x2 matrix with edit distances normalized by truth lengths.
        output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
                  [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
  """
  return _op_def_lib.apply_op("EditDistance",
                              hypothesis_indices=hypothesis_indices,
                              hypothesis_values=hypothesis_values,
                              hypothesis_shape=hypothesis_shape,
                              truth_indices=truth_indices,
                              truth_values=truth_values,
                              truth_shape=truth_shape, normalize=normalize,
                              name=name)


def expand_dims(input, dim, name=None):
  r"""Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
  zero; if you specify a negative number for `dim` it is counted backward from
  the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```prettyprint
  # 't' is a tensor of shape [2]
  shape(expand_dims(t, 0)) ==> [1, 2]
  shape(expand_dims(t, 1)) ==> [2, 1]
  shape(expand_dims(t, -1)) ==> [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    dim: A `Tensor` of type `int32`.
      0-D (scalar). Specifies the dimension index at which to
      expand the shape of `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but its shape has an additional
    dimension of size 1 added.
  """
  return _op_def_lib.apply_op("ExpandDims", input=input, dim=dim, name=name)


def fill(dims, value, name=None):
  r"""Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```prettyprint
  # output tensor shape needs to be [2, 3]
  # so 'dims' is [2, 3]
  fill(dims, 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
  ```

  Args:
    dims: A `Tensor` of type `int32`.
      1-D. Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  return _op_def_lib.apply_op("Fill", dims=dims, value=value, name=name)


def gather(params, indices, name=None):
  r"""Gather slices from `params` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

  If `indices` is a permutation and `len(indices) == params.shape[0]` then
  this operation will permute `params` accordingly.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/Gather.png" alt>
  </div>

  Args:
    params: A `Tensor`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  return _op_def_lib.apply_op("Gather", params=params, indices=indices,
                              name=name)


def identity(input, name=None):
  r"""Return a tensor with the same shape and contents as the input tensor or value.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("Identity", input=input, name=name)


def invert_permutation(x, name=None):
  r"""Computes the inverse permutation of a tensor.

  This operation computes the inverse of an index permutation. It takes a 1-D
  integer tensor `x`, which represents the indices of a zero-based array, and
  swaps each value with its index position. In other words, for an ouput tensor
  `y` and an input tensor `x`, this operation computes the following:

  `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

  The values must include 0. There can be no duplicate values or negative values.

  For example:

  ```prettyprint
  # tensor `x` is [3, 4, 0, 2, 1]
  invert_permutation(x) ==> [2, 4, 3, 0, 1]
  ```

  Args:
    x: A `Tensor` of type `int32`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. 1-D.
  """
  return _op_def_lib.apply_op("InvertPermutation", x=x, name=name)


def list_diff(x, y, name=None):
  r"""Computes the difference between two lists of numbers.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all numbers that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```prettyprint
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```prettyprint
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).
    out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
    idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
  """
  return _op_def_lib.apply_op("ListDiff", x=x, y=y, name=name)


def _pack(values, name=None):
  r"""Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the `N` tensors in `values` into a tensor with rank one higher than each
  tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
  `output[i, ...] = values[i][...]`.

  This is the opposite of `unpack`.

  Args:
    values: A list of at least 1 `Tensor` objects of the same type.
      Must be of same shape and type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`. The packed tensor.
  """
  return _op_def_lib.apply_op("Pack", values=values, name=name)


def pad(input, paddings, name=None):
  r"""Pads a tensor with zeros.

  This operation pads a `input` with zeros according to the `paddings` you
  specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
  rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many zeros to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  in that dimension.

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```prettyprint
  # 't' is [[1, 1], [2, 2]]
  # 'paddings' is [[1, 1]], [2, 2]]
  # rank of 't' is 2
  pad(t, paddings) ==> [[0, 0, 0, 0, 0]
                        [0, 0, 0, 0, 0]
                        [0, 1, 1, 0, 0]
                       [[0, 2, 2, 0, 0]
                        [0, 0, 0, 0, 0]]
  ```

  Args:
    input: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("Pad", input=input, paddings=paddings,
                              name=name)


def _placeholder(dtype, shape, name=None):
  r"""A placeholder op for a value that will be fed into the computation.

  N.B. This operation will fail with an error if it is executed. It is
  intended as a way to represent a value that will always be fed, and to
  provide attrs that enable the fed value to be checked at runtime.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`.
      (Optional) The shape of the tensor. If the shape has 0 dimensions, the
      shape is unconstrained.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A placeholder tensor that must be replaced using the feed mechanism.
  """
  return _op_def_lib.apply_op("Placeholder", dtype=dtype, shape=shape,
                              name=name)


def rank(input, name=None):
  r"""Returns the rank of a tensor.

  This operation returns an integer representing the rank of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
  of a tensor is the number of indices required to uniquely select each element
  of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  return _op_def_lib.apply_op("Rank", input=input, name=name)


def _ref_identity(input, name=None):
  r"""Return the same ref tensor as the input ref tensor.

  Args:
    input: A mutable `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("RefIdentity", input=input, name=name)


def reshape(tensor, shape, name=None):
  r"""Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If `shape` is the special value `[-1]`, then `tensor` is flattened and the
  operation outputs a 1-D tensor with all elements of `tensor`.

  If `shape` is 1-D or higher, then the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of elements
  implied by `shape` must be the same as the number of elements in `tensor`.

  For example:

  ```prettyprint
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3]
                          [4, 5, 6]
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]]
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2]
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  return _op_def_lib.apply_op("Reshape", tensor=tensor, shape=shape,
                              name=name)


def reverse(tensor, dims, name=None):
  r"""Reverses specific dimensions of a tensor.

  Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
  of `tensor`, this operation reverses each dimension i of `tensor` where
  `dims[i]` is `True`.

  `tensor` can have up to 8 dimensions. The number of dimensions
  of `tensor` must equal the number of elements in `dims`. In other words:

  `rank(tensor) = size(dims)`

  For example:

  ```prettyprint
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [False, False, False, True]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is [False, True, False, False]
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is [False, False, True, False]
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `bool`, `float32`, `float64`.
      Up to 8-D.
    dims: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.
  """
  return _op_def_lib.apply_op("Reverse", tensor=tensor, dims=dims, name=name)


def reverse_sequence(input, seq_lengths, seq_dim, name=None):
  r"""Reverses variable length slices in dimension `seq_dim`.

  This op first slices `input` along the first dimension, and for each slice `i`,
  reverses the first `seq_lengths[i]` elements along the dimension `seq_dim`.

  The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
  and `seq_lengths` must be a vector of length `input.dims(0)`.

  The output slice `i` along dimension 0 is then given by input slice `i`, with
  the first `seq_lengths[i]` slices along dimension `seq_dim` reversed.

  For example:

  ```prettyprint
  # Given this:
  seq_dim = 1
  input.dims = (4, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
  output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
  output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
  output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

  # while entries past seq_lens are copied through:
  output[0, 7:, :, ...] = input[0, 7:, :, ...]
  output[1, 2:, :, ...] = input[1, 2:, :, ...]
  output[2, 3:, :, ...] = input[2, 3:, :, ...]
  output[3, 2:, :, ...] = input[3, 2:, :, ...]
  ```

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor` of type `int64`.
      1-D with length `input.dims(0)` and
      `max(seq_lengths) < input.dims(seq_dim)`
    seq_dim: An `int`. The dimension which is partially reversed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The partially reversed input. It has the same shape as `input`.
  """
  return _op_def_lib.apply_op("ReverseSequence", input=input,
                              seq_lengths=seq_lengths, seq_dim=seq_dim,
                              name=name)


def shape(input, name=None):
  r"""Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  return _op_def_lib.apply_op("Shape", input=input, name=name)


def size(input, name=None):
  r"""Returns the size of a tensor.

  This operation returns an integer representing the number of elements in
  `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  return _op_def_lib.apply_op("Size", input=input, name=name)


def _slice(input, begin, size, name=None):
  r"""Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in
  'begin'.

  *Requirements*:
    0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      begin[i] specifies the offset into the 'i'th dimension of
      'input' to slice from.
    size: A `Tensor`. Must have the same type as `begin`.
      size[i] specifies the number of elements of the 'i'th dimension
      of 'input' to slice. If size[i] is -1, all remaining elements in dimension
      i are included in the slice (i.e. this is equivalent to setting
      size[i] = input.dim_size(i) - begin[i]).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("Slice", input=input, begin=begin, size=size,
                              name=name)


def _split(split_dim, value, num_split, name=None):
  r"""Splits a tensor into `num_split` tensors along one dimension.

  Args:
    split_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to split.  Must be in the range
      `[0, rank(value))`.
    value: A `Tensor`. The tensor to split.
    num_split: An `int` that is `>= 1`.
      The number of ways to split.  Must evenly divide
      `value.shape[split_dim]`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_split` `Tensor` objects of the same type as value.
    They are identically shaped tensors, whose shape matches that of `value`
    except along `split_dim`, where their sizes are
    `values.shape[split_dim] / num_split`.
  """
  return _op_def_lib.apply_op("Split", split_dim=split_dim, value=value,
                              num_split=num_split, name=name)


def squeeze(input, squeeze_dims=None, name=None):
  r"""Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `squeeze_dims`.

  For example:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t)) ==> [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
  ```

  Args:
    input: A `Tensor`. The `input` to squeeze.
    squeeze_dims: An optional list of `ints`. Defaults to `[]`.
      If specified, only squeezes the dimensions listed. The dimension
      index starts at 0. It is an error to squeeze a dimension that is not 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.
  """
  return _op_def_lib.apply_op("Squeeze", input=input,
                              squeeze_dims=squeeze_dims, name=name)


def stop_gradient(input, name=None):
  r"""Stops gradient computation.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, this op prevents the contribution of
  its inputs to be taken into account.  Normally, the gradient generator adds ops
  to a graph to compute the derivatives of a specified 'loss' by recursively
  finding out inputs that contributed to its computation.  If you insert this op
  in the graph it inputs are masked from the gradient generator.  They are not
  taken into account for computing gradients.

  This is useful any time you want to compute a value with TensorFlow but need
  to pretend that the value was a constant. Some examples include:

  *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
  *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
  *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("StopGradient", input=input, name=name)


def tile(input, multiples, name=None):
  r"""Constructs a tensor by tiling a given tensor.

  This operation creates a new tensor by replicating `input` `multiples` times.
  The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  and the values of `input` are replicated `multiples[i]` times along the 'i'th
  dimension. For example, tiling `[a b c d]` by `[2]` produces
  `[a b c d a b c d]`.

  Args:
    input: A `Tensor`. 1-D or higher.
    multiples: A `Tensor` of type `int32`.
      1-D. Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("Tile", input=input, multiples=multiples,
                              name=name)


def _tile_grad(input, multiples, name=None):
  r"""Returns the gradient of `Tile`.

  Since `Tile` takes an input and repeats the input `multiples` times
  along each dimension, `TileGrad` takes in `multiples` and aggregates
  each repeated tile of `input` into `output`.

  Args:
    input: A `Tensor`.
    multiples: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return _op_def_lib.apply_op("TileGrad", input=input, multiples=multiples,
                              name=name)


def transpose(x, perm, name=None):
  r"""Shuffle dimensions of x according to a permutation.

  The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

  Args:
    x: A `Tensor`.
    perm: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _op_def_lib.apply_op("Transpose", x=x, perm=perm, name=name)


def unique(x, name=None):
  r"""Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```prettyprint
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  Args:
    x: A `Tensor`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).
    y: A `Tensor`. Has the same type as `x`. 1-D.
    idx: A `Tensor` of type `int32`. 1-D.
  """
  return _op_def_lib.apply_op("Unique", x=x, name=name)


def _unpack(value, num, name=None):
  r"""Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the first dimension.
  The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
  `output` has shape `value.shape[1:]`.

  This is the opposite of `pack`.

  Args:
    value: A `Tensor`. 1-D or higher, with first dimension `num`.
    num: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A list of `num` `Tensor` objects of the same type as value.
    The list of tensors unpacked from `value`.
  """
  return _op_def_lib.apply_op("Unpack", value=value, num=num, name=name)


def where(input, name=None):
  r"""Returns locations of true values in a boolean tensor.

  This operation returns the coordinates of true elements in `input`. The
  coordinates are returned in a 2-D tensor where the first dimension (rows)
  represents the number of true elements, and the second dimension (columns)
  represents the coordinates of the true elements. Keep in mind, the shape of
  the output tensor can vary depending on how many true values there are in
  `input`. Indices are output in row-major order.

  For example:

  ```prettyprint
  # 'input' tensor is [[True, False]
  #                    [True, False]]
  # 'input' has two true values, so output has two coordinates.
  # 'input' has rank of 2, so coordinates have two indices.
  where(input) ==> [[0, 0],
                    [1, 0]]

  # `input` tensor is [[[True, False]
  #                     [True, False]]
  #                    [[False, True]
  #                     [False, True]]
  #                    [[False, False]
  #                     [False, True]]]
  # 'input' has 5 true values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]
  ```

  Args:
    input: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  return _op_def_lib.apply_op("Where", input=input, name=name)


def _zeros_like(x, name=None):
  r"""Returns a tensor of zeros with the same shape and type as x.

  Args:
    x: A `Tensor`. a tensor of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
    a tensor of the same shape and type as x but filled with zeros.
  """
  return _op_def_lib.apply_op("ZerosLike", x=x, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BroadcastGradientArgs"
  input_arg {
    name: "s0"
    type: DT_INT32
  }
  input_arg {
    name: "s1"
    type: DT_INT32
  }
  output_arg {
    name: "r0"
    type: DT_INT32
  }
  output_arg {
    name: "r1"
    type: DT_INT32
  }
}
op {
  name: "CheckNumerics"
  input_arg {
    name: "tensor"
    type_attr: "T"
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
  attr {
    name: "message"
    type: "string"
  }
}
op {
  name: "Concat"
  input_arg {
    name: "concat_dim"
    type: DT_INT32
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 2
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Const"
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "value"
    type: "tensor"
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "Diag"
  input_arg {
    name: "diagonal"
    type_attr: "T"
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
      }
    }
  }
}
op {
  name: "EditDistance"
  input_arg {
    name: "hypothesis_indices"
    type: DT_INT64
  }
  input_arg {
    name: "hypothesis_values"
    type_attr: "T"
  }
  input_arg {
    name: "hypothesis_shape"
    type: DT_INT64
  }
  input_arg {
    name: "truth_indices"
    type: DT_INT64
  }
  input_arg {
    name: "truth_values"
    type_attr: "T"
  }
  input_arg {
    name: "truth_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "normalize"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "ExpandDims"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dim"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Fill"
  input_arg {
    name: "dims"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Gather"
  input_arg {
    name: "params"
    type_attr: "Tparams"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "output"
    type_attr: "Tparams"
  }
  attr {
    name: "Tparams"
    type: "type"
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
  name: "Identity"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "InvertPermutation"
  input_arg {
    name: "x"
    type: DT_INT32
  }
  output_arg {
    name: "y"
    type: DT_INT32
  }
}
op {
  name: "ListDiff"
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
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Pack"
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
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
  }
}
op {
  name: "Pad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Placeholder"
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "shape"
    type: "shape"
  }
}
op {
  name: "Rank"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "RefIdentity"
  input_arg {
    name: "input"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Reshape"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  input_arg {
    name: "shape"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Reverse"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  input_arg {
    name: "dims"
    type: DT_BOOL
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
        type: DT_INT32
        type: DT_BOOL
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "ReverseSequence"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "seq_lengths"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "seq_dim"
    type: "int"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Shape"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Size"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Slice"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "begin"
    type_attr: "Index"
  }
  input_arg {
    name: "size"
    type_attr: "Index"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "Index"
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
  name: "Split"
  input_arg {
    name: "split_dim"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
    number_attr: "num_split"
  }
  attr {
    name: "num_split"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Squeeze"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "squeeze_dims"
    type: "list(int)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
}
op {
  name: "StopGradient"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Tile"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "multiples"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TileGrad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "multiples"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Transpose"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "perm"
    type: DT_INT32
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Unique"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Unpack"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
    number_attr: "num"
  }
  attr {
    name: "num"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Where"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  output_arg {
    name: "index"
    type: DT_INT64
  }
}
op {
  name: "ZerosLike"
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
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
