"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def dynamic_partition(data, partitions, num_partitions, name=None):
  r"""Partitions `data` into `num_partitions` tensors using indices from `partitions`.

  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
  are placed in `outputs[i]` in lexicographic order of `js`, and the first
  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
  In detail,

      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

  `data.shape` must start with `partitions.shape`.

  For example:

      # Scalar partitions
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/DynamicPartition.png" alt>
  </div>

  Args:
    data: A `Tensor`.
    partitions: A `Tensor` of type `int32`.
      Any shape.  Indices in the range `[0, num_partitions)`.
    num_partitions: An `int` that is `>= 1`.
      The number of partitions to output.
    name: A name for the operation (optional).

  Returns:
    A list of `num_partitions` `Tensor` objects of the same type as data.
  """
  return _op_def_lib.apply_op("DynamicPartition", data=data,
                              partitions=partitions,
                              num_partitions=num_partitions, name=name)


def dynamic_stitch(indices, data, name=None):
  r"""Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

  For example, if each `indices[m]` is scalar or vector, we have

      # Scalar indices
      merged[indices[m], ...] = data[m][...]

      # Vector indices
      merged[indices[m][i], ...] = data[m][i, ...]

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result.

  For example:

      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int32`.
    data: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  return _op_def_lib.apply_op("DynamicStitch", indices=indices, data=data,
                              name=name)


def _fifo_queue(component_types, shapes=None, capacity=None, container=None,
                shared_name=None, name=None):
  r"""A queue that produces elements in first-in first-out order.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  return _op_def_lib.apply_op("FIFOQueue", component_types=component_types,
                              shapes=shapes, capacity=capacity,
                              container=container, shared_name=shared_name,
                              name=name)


def _hash_table(key_dtype, value_dtype, container=None, shared_name=None,
                name=None):
  r"""Creates and holds an immutable hash table.

  The key and value types can be specified. After initialization, the table
  becomes immutable.

  Args:
    key_dtype: A `tf.DType`. the type of the table key.
    value_dtype: A `tf.DType`. the type of the table value.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this hash table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this hash table is shared under the given name across
      multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. a handle of a the lookup table.
  """
  return _op_def_lib.apply_op("HashTable", key_dtype=key_dtype,
                              value_dtype=value_dtype, container=container,
                              shared_name=shared_name, name=name)


def _initialize_table(table_handle, keys, values, name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      a handle of the lookup table to be initialized.
    keys: A `Tensor`. a vector of keys of type Tkey.
    values: A `Tensor`. a vector of values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  return _op_def_lib.apply_op("InitializeTable", table_handle=table_handle,
                              keys=keys, values=values, name=name)


def _lookup_table_find(table_handle, input_values, default_value, name=None):
  r"""Maps elements of a tensor into associated values given a lookup table.

  If an element of the input_values is not present in the table, the
  specified default_value is used.

  The table needs to be initialized and the input and output types correspond
  to the table key and value types.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      A handle for a lookup table.
    input_values: A `Tensor`. A vector of key values.
    default_value: A `Tensor`.
      A scalar to return if the input is not found in the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
    A vector of values associated to the inputs.
  """
  return _op_def_lib.apply_op("LookupTableFind", table_handle=table_handle,
                              input_values=input_values,
                              default_value=default_value, name=name)


def _lookup_table_size(table_handle, name=None):
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      The handle to a lookup table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`. The number of elements in the given table.
  """
  return _op_def_lib.apply_op("LookupTableSize", table_handle=table_handle,
                              name=name)


def _queue_close(handle, cancel_pending_enqueues=None, name=None):
  r"""Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be cancelled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  return _op_def_lib.apply_op("QueueClose", handle=handle,
                              cancel_pending_enqueues=cancel_pending_enqueues,
                              name=name)


def _queue_dequeue(handle, component_types, timeout_ms=None, name=None):
  r"""Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  return _op_def_lib.apply_op("QueueDequeue", handle=handle,
                              component_types=component_types,
                              timeout_ms=timeout_ms, name=name)


def _queue_dequeue_many(handle, n, component_types, timeout_ms=None,
                        name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has k outputs, where k is the number of components in
  the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until n elements
  have been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  return _op_def_lib.apply_op("QueueDequeueMany", handle=handle, n=n,
                              component_types=component_types,
                              timeout_ms=timeout_ms, name=name)


def _queue_enqueue(handle, components, timeout_ms=None, name=None):
  r"""Enqueues a tuple of one or more tensors in the given queue.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  element has been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is full, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  return _op_def_lib.apply_op("QueueEnqueue", handle=handle,
                              components=components, timeout_ms=timeout_ms,
                              name=name)


def _queue_enqueue_many(handle, components, timeout_ms=None, name=None):
  r"""Enqueues zero or more tuples of one or more tensors in the given queue.

  This operation slices each component tensor along the 0th dimension to
  make multiple queue elements. All of the tuple components must have the
  same size in the 0th dimension.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  elements have been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should
      be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is too full, this operation will block for up
      to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  return _op_def_lib.apply_op("QueueEnqueueMany", handle=handle,
                              components=components, timeout_ms=timeout_ms,
                              name=name)


def _queue_size(handle, name=None):
  r"""Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The number of elements in the given queue.
  """
  return _op_def_lib.apply_op("QueueSize", handle=handle, name=name)


def _random_shuffle_queue(component_types, shapes=None, capacity=None,
                          min_after_dequeue=None, seed=None, seed2=None,
                          container=None, shared_name=None, name=None):
  r"""A queue that randomizes the order of elements.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    min_after_dequeue: An optional `int`. Defaults to `0`.
      Dequeue will block unless there would be this
      many elements after the dequeue or the queue is closed. This
      ensures a minimum level of mixing of elements.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  return _op_def_lib.apply_op("RandomShuffleQueue",
                              component_types=component_types, shapes=shapes,
                              capacity=capacity,
                              min_after_dequeue=min_after_dequeue, seed=seed,
                              seed2=seed2, container=container,
                              shared_name=shared_name, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DynamicPartition"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "partitions"
    type: DT_INT32
  }
  output_arg {
    name: "outputs"
    type_attr: "T"
    number_attr: "num_partitions"
  }
  attr {
    name: "num_partitions"
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
  name: "DynamicStitch"
  input_arg {
    name: "indices"
    type: DT_INT32
    number_attr: "N"
  }
  input_arg {
    name: "data"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "merged"
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
  name: "FIFOQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "HashTable"
  output_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
}
op {
  name: "InitializeTable"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tkey"
  }
  input_arg {
    name: "values"
    type_attr: "Tval"
  }
  attr {
    name: "Tkey"
    type: "type"
  }
  attr {
    name: "Tval"
    type: "type"
  }
}
op {
  name: "LookupTableFind"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "input_values"
    type_attr: "Tin"
  }
  input_arg {
    name: "default_value"
    type_attr: "Tout"
  }
  output_arg {
    name: "output_values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableSize"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT64
  }
}
op {
  name: "QueueClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "cancel_pending_enqueues"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "QueueDequeue"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "n"
    type: DT_INT32
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueue"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueueMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueSize"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "RandomShuffleQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "min_after_dequeue"
    type: "int"
    default_value {
      i: 0
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
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
