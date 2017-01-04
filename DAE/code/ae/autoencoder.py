from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from utils.data import fill_feed_dict_ae, read_unlabeled_data
from utils.flags import FLAGS
from utils.eval import loss_supervised, evaluation, do_eval_summary
from utils.utils import tile_raster_images


class AutoEncoder(object):
  """Generic deep autoencoder.

  Autoencoder used for full training cycle, including
  unsupervised pretraining layers and final fine tuning.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final output
  logits.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess):
    """Autoencoder initializer

    Args:
      shape: list of ints specifying
              num input, hidden1 units,...hidden_n units, num logits
      sess: tensorflow session object to use
    """
    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
    self.__num_hidden_layers = len(self.__shape) - 2

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def shape(self):
    return self.__shape

  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def session(self):
    return self.__sess

  def __getitem__(self, item):
    """Get autoencoder tf variable

    Returns the specified variable created by this object.
    Names are weights#, biases#, biases#_out, weights#_fixed,
    biases#_fixed.

    Args:
     item: string, variables internal name
    Returns:
     Tensorflow variable
    """
    return self.__variables[item]

  def __setitem__(self, key, value):
    """Store a tensorflow variable

    NOTE: Don't call this explicity. It should
    be used only internally when setting up
    variables.

    Args:
      key: string, name of variable
      value: tensorflow variable
    """
    self.__variables[key] = value

  def _setup_variables(self):
    with tf.name_scope("autoencoder_variables"):
      for i in xrange(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.mul(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                   name=name_w,
                                   trainable=True)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        if i <= self.__num_hidden_layers:
          
          # Hidden layer fixed weights (after pretraining before fine tuning)
          self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                name=name_w + "_fixed",
                                                trainable=False)

          # Hidden layer fixed biases
          self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                name=name_b + "_fixed",
                                                trainable=False)

          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,
                                         trainable=True,
                                         name=name_b_out)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables_to_init(self, n):
    """Return variables that need initialization

    This method aides in the initialization of variables
    before training begins at step n. The returned
    list should be than used as the input to
    tf.initialize_variables

    Args:
      n: int giving step of training
    """
    assert n > 0
    assert n <= self.__num_hidden_layers + 1

    vars_to_init = [self._w(n), self._b(n)]

    if n <= self.__num_hidden_layers:
      vars_to_init.append(self._b(n, "_out"))

    if 1 < n <= self.__num_hidden_layers+1:
      # Fixed matrices for learning of deeper layers
      vars_to_init.append(self._w(n - 1, "_fixed"))
      vars_to_init.append(self._b(n - 1, "_fixed"))

    return vars_to_init

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def pretrain_net(self, input_pl, n, dropout, is_target=False):
    """Return net for step n training or target net

    Args:
      input_pl:  tensorflow placeholder of AE inputs
      n:         int specifying pretrain step
      is_target: bool specifying if required tensor
                  should be the target tensor
    Returns:
      Tensor giving pretraining net or pretraining target
    """

    assert n > 0
    assert n <= self.__num_hidden_layers+1

    #First - Apply Dropout
    last_output = tf.nn.dropout(input_pl, dropout)
    
    # Then use fixed weights for all the previous layers
    for i in xrange(n - 1):
      w = self._w(i + 1, "_fixed")
      b = self._b(i + 1, "_fixed")

      last_output = self._activate(last_output, w, b)


    # return input as a target, if needed
    if is_target:
      return last_output

    last_output = self._activate(last_output, self._w(n), self._b(n))

    # check if we train an output layer
    if(n== self.__num_hidden_layers+1):
      return last_output

    out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True)

    # restrict outout to the range [0,1], with a precition
    out = tf.maximum(out, 1.e-9)
    out = tf.minimum(out, 1 - 1.e-9)
    return out

  def supervised_net(self, input_pl):
    """Get the supervised fine tuning net

    Args:
      input_pl: tf placeholder for ae input data
    Returns:
      Tensor giving full ae net
    """
    last_output = input_pl

    for i in xrange(self.__num_hidden_layers+1): # +1 ?
      # Fine tuning will be done on these variables
      w = self._w(i + 1)
      b = self._b(i + 1)

      last_output = self._activate(last_output, w, b)

    return last_output


loss_summaries = {}


def training(optimizer, global_step,loss, learning_rate, loss_key=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training. """

  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
  else:
    tf.summary.scalar(loss.op.name, loss)
  
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def loss_reconstruction(output, target):
  """ Reconstruction error

  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of mean eucledean distance
  """
  with tf.name_scope("reconstruction_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      # Euclidean distance between net_output_tf,target_tf
      l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(net_output_tf, target_tf)),
                                    reduction_indices=1)) # do we need a square root here?
      return tf.reduce_mean(l2diff,reduction_indices=0)

def main_unsupervised():
  with tf.Graph().as_default() as g:
    sess = tf.Session()

    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in xrange(num_hidden)]
    ae_shape = [FLAGS.DoF] + ae_hidden_shapes + [FLAGS.DoF]

    print('AE structure is ', ae_shape)

    ae = AutoEncoder(ae_shape, sess)

    data = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders)

    print('We have ', data.train.num_sequences, ' train sequences')
    
    num_train_seq = data.train.num_sequences

    learning_rates = {j: getattr(FLAGS,
                                 "pre_layer{0}_learning_rate".format(j + 1))
                      for j in xrange(num_hidden+1)}

    variance = {j: getattr(FLAGS, "variance_{0}".format(j + 1))
             for j in xrange(num_hidden+1)}

    keep_prob = tf.placeholder(tf.float32) #dropout placeholder
    dropout = FLAGS.dropout # (keep probability) value

    variables_to_save = []


    # Train matrices between hidden layers
    for i in xrange(len(ae_shape) - 2):
      n = i + 1
      with tf.variable_scope("pretrain_{0}".format(n)):

        input_ = tf.placeholder(dtype=tf.float32,
                                shape=(None, ae_shape[0]),
                                name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(None, ae_shape[0]),
                                 name='ae_target_pl')
        
        layer = ae.pretrain_net(input_,  n, dropout)

        with tf.name_scope("target"):
          target_for_loss = ae.pretrain_net(target_, n, dropout, is_target=True)

        with tf.name_scope("loss"):
          loss = loss_reconstruction(layer, target_for_loss)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # create an optimizer
        train_op = tf.train.GradientDescentOptimizer(learning_rates[i]).minimize(loss, global_step=global_step) #TODO: change to Adam

        #train_op = tf.train.AdamOptimizer(learning_rate=learning_rates[i]).minimize(loss)
        #adamOptimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

        print('Optimizer was created!')
        
        

    
        #Initialize variables
        vars_to_init = ae.get_variables_to_init(n)
        vars_to_init.append(global_step)
        vars_to_init.append

        sess.run(tf.variables_initializer(vars_to_init))
        # TODO: change to the following line
        # sess.run(tf.initialize_all_variables())
        
        variables_to_save.append(ae.get_variables_to_init( n))

        # Prepare for making a summary for TensorBoard

        summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(n))

        summary_writer = tf.summary.FileWriter(summary_dir,
                                               graph=sess.graph,
                                               flush_secs=FLAGS.flush_secs)

        loss_summary = tf.summary.scalar('Average_reconstruction_error', loss)

        summary_op = tf.summary.merge([loss_summary])

        print("\n\n")
        print("| Tr. Sequence | Error    |  Layer  |   Epoch  |")
        print("|--------------|----------|---------|----------|")

        for step in xrange(FLAGS.pretraining_epochs * num_train_seq):
          feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance[i], dropout)

          loss_summary, loss_value = sess.run([train_op, loss],
                                              feed_dict=feed_dict)

          if(step%100 == 0):
            # Write summary
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
              
            # Print results of screen
            output = "| {0:>12} | {1:8.4f} | Layer {2} | Epoch {3}  |"\
                       .format(step, loss_value, n, step // num_train_seq + 1)

            print(output)

        summary_writer.flush()

    # Train matrix between the last hidden layer and output layer
    i = len(ae_shape)-2
    n = i + 1

    dropout=1 # no dropout for the last layer training and testing
    
    with tf.variable_scope("pretrain_{0}".format(n)):

      input_ = tf.placeholder(dtype=tf.float32,
                                shape=(None, ae_shape[0]),
                                name='ae_input_pl')
      target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(None, ae_shape[0]),
                                 name='ae_target_pl')
      
      output_layer = ae.pretrain_net(input_, n, dropout)

      loss = loss_reconstruction(output_layer, target_)

      test_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')

      loss_summary = tf.summary.scalar('Average_reconstruction_error', loss)
      train_summary_op = tf.summary.merge([loss_summary])
      test_summary_op =  tf.summary.scalar('Test_reconstr_error',test_error)

      tr_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_train')
      tr_summary_writer = tf.summary.FileWriter(tr_summary_dir)
      test_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_test')
      test_summary_writer = tf.summary.FileWriter(test_summary_dir)
      

      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
    
      #Initialize variables
      vars_to_init = ae.get_variables_to_init(n)
      vars_to_init.append(global_step)
      vars_to_init.append

      sess.run(tf.variables_initializer(vars_to_init))
        
      variables_to_save.append(ae.get_variables_to_init( n))

      # create an optimizer
      train_op = tf.train.GradientDescentOptimizer(learning_rates[i]).minimize(loss, global_step=global_step)
      #train_op = training(adamOptimizer,global_step,loss, learning_rates[i], i)

      variables_to_save.append(ae.get_variables_to_init(n))

      print("\n\nTesting after each 100th sequence of training of the last layer:\n\n")

      for step in xrange(FLAGS.last_layer_epochs * num_train_seq):

        # Train
        feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance[i], dropout)
        loss_summary, loss_value = sess.run([train_op, loss],
                                              feed_dict=feed_dict)


        if(step%100 == 0):
          # Update the events file.
          train_sum = sess.run(train_summary_op, feed_dict=feed_dict)
          tr_summary_writer.add_summary(train_sum, step)

          #Evaluate on the test sequences
          error_sum=0
          num_test_seq = data.test.num_sequences
          for test_seq in range(num_test_seq):
            feed_dict = fill_feed_dict_ae(data.test, input_, target_, keep_prob, variance[i], dropout)
            curr_err = sess.run(loss, feed_dict=feed_dict)
            error_sum+= curr_err
            test_error_ = error_sum/num_test_seq
          test_sum = sess.run(test_summary_op, feed_dict={test_error: test_error_})
          test_summary_writer.add_summary(test_sum, step)

          # Print result to stdout.
          print('Step %d: loss = %.2f' % (step, test_error_))
          

      summary_writer.flush()

    # Create a saver
    saver = tf.train.Saver(np.concatenate(variables_to_save, axis=0).tolist())

    # Save a model
    saver.save(sess,FLAGS.params_file) #TODO : do we need it?
  
  return ae
    
 

if __name__ == '__main__':
  ae = main_unsupervised()
  # main_supervised(ae)
