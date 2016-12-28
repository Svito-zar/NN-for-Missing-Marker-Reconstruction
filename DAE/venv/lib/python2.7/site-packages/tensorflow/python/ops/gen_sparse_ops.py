"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _sparse_concat(indices, values, shapes, concat_dim, name=None):
  r"""Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of these sparse tensors.
  It is assumed that each input is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
      1-D.  Non-empty values of each `SparseTensor`.
    shapes: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of type `int64`.
      1-D.  Shapes of each `SparseTensor`.
    concat_dim: An `int` that is `>= 0`. Dimension to concatenate along.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A `Tensor` of type `int64`. 2-D.  Indices of the concatenated `SparseTensor`.
    output_values: A `Tensor`. Has the same type as `values`. 1-D.  Non-empty values of the concatenated `SparseTensor`.
    output_shape: A `Tensor` of type `int64`. 1-D.  Shape of the concatenated `SparseTensor`.
  """
  return _op_def_lib.apply_op("SparseConcat", indices=indices, values=values,
                              shapes=shapes, concat_dim=concat_dim, name=name)


def _sparse_reorder(input_indices, input_values, input_shape, name=None):
  r"""Reorders a SparseTensor into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering along
  increasing dimension number. The only time ordering can be violated is during
  manual manipulation of the indices and values vectors to add entries.

  Reordering does not affect the shape of the SparseTensor.

  If the tensor has rank `R` and `N` non-empty values, `input_indices` has
  shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).
    output_indices: A `Tensor` of type `int64`. 2-D.  `N x R` matrix with the same indices as input_indices, but
      in canonical row-major ordering.
    output_values: A `Tensor`. Has the same type as `input_values`. 1-D.  `N` non-empty values corresponding to `output_indices`.
  """
  return _op_def_lib.apply_op("SparseReorder", input_indices=input_indices,
                              input_values=input_values,
                              input_shape=input_shape, name=name)


def sparse_to_dense(sparse_indices, output_shape, sparse_values,
                    default_value, name=None):
  r"""Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```prettyprint
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values` is a
  scalar, all sparse indices are set to this single value.

  Args:
    sparse_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
      index where `sparse_values[i]` will be placed.
    output_shape: A `Tensor`. Must have the same type as `sparse_indices`.
      1-D.  Shape of the dense output tensor.
    sparse_values: A `Tensor`.
      1-D.  Values corresponding to each row of `sparse_indices`,
      or a scalar value to be used for all sparse indices.
    default_value: A `Tensor`. Must have the same type as `sparse_values`.
      Scalar value to set for indices not specified in
      `sparse_indices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sparse_values`.
    Dense output tensor of shape `output_shape`.
  """
  return _op_def_lib.apply_op("SparseToDense", sparse_indices=sparse_indices,
                              output_shape=output_shape,
                              sparse_values=sparse_values,
                              default_value=default_value, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "SparseConcat"
  input_arg {
    name: "indices"
    type: DT_INT64
    number_attr: "N"
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  input_arg {
    name: "shapes"
    type: DT_INT64
    number_attr: "N"
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
  }
  attr {
    name: "concat_dim"
    type: "int"
    has_minimum: true
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
  name: "SparseReorder"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type_attr: "T"
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseToDense"
  input_arg {
    name: "sparse_indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "output_shape"
    type_attr: "Tindices"
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "default_value"
    type_attr: "T"
  }
  output_arg {
    name: "dense"
    type_attr: "T"
  }
  attr {
    name: "T"
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
"""


_op_def_lib = _InitOpDefLibrary()
