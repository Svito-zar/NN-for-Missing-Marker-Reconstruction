"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _assert(condition, data, summarize=None, name=None):
  r"""Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: A `Tensor` of type `bool`. The condition to evaluate.
    data: A list of `Tensor` objects.
      The tensors to print out when condition is false.
    summarize: An optional `int`. Defaults to `3`.
      Print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  return _op_def_lib.apply_op("Assert", condition=condition, data=data,
                              summarize=summarize, name=name)


def _print(input, data, message=None, first_n=None, summarize=None,
           name=None):
  r"""Prints a list of tensors.

  Passes `input` through to `output` and prints `data` when evaluating.

  Args:
    input: A `Tensor`. The tensor passed to `output`
    data: A list of `Tensor` objects.
      A list of tensors to print out when op is evaluated.
    message: An optional `string`. Defaults to `""`.
      A string, prefix of the error message.
    first_n: An optional `int`. Defaults to `-1`.
      Only log `first_n` number of times. -1 disables logging.
    summarize: An optional `int`. Defaults to `3`.
      Only print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The unmodified `input` tensor
  """
  return _op_def_lib.apply_op("Print", input=input, data=data,
                              message=message, first_n=first_n,
                              summarize=summarize, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Assert"
  input_arg {
    name: "condition"
    type: DT_BOOL
  }
  input_arg {
    name: "data"
    type_list_attr: "T"
  }
  attr {
    name: "T"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
}
op {
  name: "Print"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "data"
    type_list_attr: "U"
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
    name: "U"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "message"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "first_n"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
