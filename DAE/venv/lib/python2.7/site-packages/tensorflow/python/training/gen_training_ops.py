"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def apply_adagrad(var, accum, lr, grad, use_locking=None, name=None):
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("ApplyAdagrad", var=var, accum=accum, lr=lr,
                              grad=grad, use_locking=use_locking, name=name)


def apply_adam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
               grad, use_locking=None, name=None):
  r"""Update '*var' according to the Adam algorithm.

  lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
  m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
  v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
  variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    v: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    beta1_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, m, and v tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("ApplyAdam", var=var, m=m, v=v,
                              beta1_power=beta1_power,
                              beta2_power=beta2_power, lr=lr, beta1=beta1,
                              beta2=beta2, epsilon=epsilon, grad=grad,
                              use_locking=use_locking, name=name)


def apply_gradient_descent(var, alpha, delta, use_locking=None, name=None):
  r"""Update '*var' by subtracting 'alpha' * 'delta' from it.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `var`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("ApplyGradientDescent", var=var, alpha=alpha,
                              delta=delta, use_locking=use_locking, name=name)


def apply_momentum(var, accum, lr, grad, momentum, use_locking=None,
                   name=None):
  r"""Update '*var' according to the momentum scheme.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("ApplyMomentum", var=var, accum=accum, lr=lr,
                              grad=grad, momentum=momentum,
                              use_locking=use_locking, name=name)


def apply_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad,
                   use_locking=None, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, m, and v tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("ApplyRMSProp", var=var, ms=ms, mom=mom, lr=lr,
                              rho=rho, momentum=momentum, epsilon=epsilon,
                              grad=grad, use_locking=use_locking, name=name)


def sparse_apply_adagrad(var, accum, lr, grad, indices, use_locking=None,
                         name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("SparseApplyAdagrad", var=var, accum=accum,
                              lr=lr, grad=grad, indices=indices,
                              use_locking=use_locking, name=name)


def sparse_apply_momentum(var, accum, lr, grad, indices, momentum,
                          use_locking=None, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  return _op_def_lib.apply_op("SparseApplyMomentum", var=var, accum=accum,
                              lr=lr, grad=grad, indices=indices,
                              momentum=momentum, use_locking=use_locking,
                              name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "ApplyAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyAdam"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "m"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "v"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "beta1_power"
    type_attr: "T"
  }
  input_arg {
    name: "beta2_power"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "beta1"
    type_attr: "T"
  }
  input_arg {
    name: "beta2"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyGradientDescent"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "delta"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyMomentum"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyRMSProp"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "ms"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mom"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyMomentum"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
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
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
