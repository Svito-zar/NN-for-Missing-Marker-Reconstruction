"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def decode_csv(records, record_defaults, field_delim=None, name=None):
  r"""Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or empty if the column is required.
    field_delim: An optional `string`. Defaults to `","`.
      delimiter to separate fields in a record.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.
  """
  return _op_def_lib.apply_op("DecodeCSV", records=records,
                              record_defaults=record_defaults,
                              field_delim=field_delim, name=name)


def decode_raw(bytes, out_type, little_endian=None, name=None):
  r"""Reinterpret the bytes of a string as a vector of numbers.

  Args:
    bytes: A `Tensor` of type `string`.
      All the elements must have the same length.
    out_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input bytes are in little-endian order.
      Ignored for out_types that are stored in a single byte like uint8.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor with one more dimension than the input bytes.  The
    added dimension will have size equal to the length of the elements
    of bytes divided by the number of bytes to represent out_type.
  """
  return _op_def_lib.apply_op("DecodeRaw", bytes=bytes, out_type=out_type,
                              little_endian=little_endian, name=name)


def _parse_example(serialized, names, sparse_keys, dense_keys, dense_defaults,
                   sparse_types, dense_shapes, name=None):
  r"""Transforms a vector of brain.Example protos (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A vector containing a batch of binary serialized Example protos.
    names: A `Tensor` of type `string`.
      A vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A list of `Tensor` objects of type `string`.
      A list of Nsparse string Tensors (scalars).
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A list of `Tensor` objects of type `string`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples' features associated with dense values.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ndense Tensors (some may be empty).
      dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, its shape must match dense_shapes[j].
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of Nsparse types; the data types of data in each Feature
      given in sparse_keys.
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      A list of Ndense shapes; the shapes of data in each Feature
      given in dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always equal dense_shapes[j].NumEntries().
      If dense_shapes[j] == (D0, D1, ..., DN) then the the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values).
    sparse_indices: A list with the same number of `Tensor` objects as `sparse_keys` of `Tensor` objects of type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list with the same number of `Tensor` objects as `sparse_keys` of `Tensor` objects of type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
  """
  return _op_def_lib.apply_op("ParseExample", serialized=serialized,
                              names=names, sparse_keys=sparse_keys,
                              dense_keys=dense_keys,
                              dense_defaults=dense_defaults,
                              sparse_types=sparse_types,
                              dense_shapes=dense_shapes, name=name)


def string_to_number(string_tensor, out_type=None, name=None):
  r"""Converts each string in the input Tensor to the specified numeric type.

  (Note that int32 overflow results in an error while float overflow
  results in a rounded value.)

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.int32`. Defaults to `tf.float32`.
      The numeric type to interpret each string in string_tensor as.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor of the same shape as the input string_tensor.
  """
  return _op_def_lib.apply_op("StringToNumber", string_tensor=string_tensor,
                              out_type=out_type, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DecodeCSV"
  input_arg {
    name: "records"
    type: DT_STRING
  }
  input_arg {
    name: "record_defaults"
    type_list_attr: "OUT_TYPE"
  }
  output_arg {
    name: "output"
    type_list_attr: "OUT_TYPE"
  }
  attr {
    name: "OUT_TYPE"
    type: "list(type)"
    has_minimum: true
    minimum: 1
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT32
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "field_delim"
    type: "string"
    default_value {
      s: ","
    }
  }
}
op {
  name: "DecodeRaw"
  input_arg {
    name: "bytes"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_INT64
      }
    }
  }
  attr {
    name: "little_endian"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "ParseExample"
  input_arg {
    name: "serialized"
    type: DT_STRING
  }
  input_arg {
    name: "names"
    type: DT_STRING
  }
  input_arg {
    name: "sparse_keys"
    type: DT_STRING
    number_attr: "Nsparse"
  }
  input_arg {
    name: "dense_keys"
    type: DT_STRING
    number_attr: "Ndense"
  }
  input_arg {
    name: "dense_defaults"
    type_list_attr: "Tdense"
  }
  output_arg {
    name: "sparse_indices"
    type: DT_INT64
    number_attr: "Nsparse"
  }
  output_arg {
    name: "sparse_values"
    type_list_attr: "sparse_types"
  }
  output_arg {
    name: "sparse_shapes"
    type: DT_INT64
    number_attr: "Nsparse"
  }
  output_arg {
    name: "dense_values"
    type_list_attr: "Tdense"
  }
  attr {
    name: "Nsparse"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "Ndense"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "sparse_types"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "Tdense"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "dense_shapes"
    type: "list(shape)"
    has_minimum: true
  }
}
op {
  name: "StringToNumber"
  input_arg {
    name: "string_tensor"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT32
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
