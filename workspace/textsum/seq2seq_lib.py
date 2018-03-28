# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""seq2seq library codes copied from elsewhere for customization."""

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_ops import *


# Adapted to support sampled_softmax loss function, which accepts activations
# instead of logits.
def sequence_loss_by_example(inputs, targets, weights, loss_function,
                             average_across_timesteps=True, name=None):
    """Sampled softmax loss for a sequence of inputs (per example).

    Args:
      inputs: List of 2D Tensors of shape [batch_size x hid_dim].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      loss_function: Sampled softmax function (inputs, labels) -> loss
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      name: Optional name for this operation, default: 'sequence_loss_by_example'.

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(inputs) is different from len(targets) or len(weights).
    """
    if len(targets) != len(inputs) or len(weights) != len(inputs):
        raise ValueError('Lengths of logits, weights, and targets must be the same '
                         '%d, %d, %d.' % (len(inputs), len(weights), len(targets)))
    with tf.op_scope(inputs + targets + weights, name,
                     'sequence_loss_by_example'):
        log_perp_list = []
        for inp, target, weight in zip(inputs, targets, weights):
            crossent = loss_function(inp, target)
            log_perp_list.append(crossent * weight)
        log_perps = tf.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sampled_sequence_loss(inputs, targets, weights, loss_function,
                          average_across_timesteps=True,
                          average_across_batch=True, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      inputs: List of 2D Tensors of shape [batch_size x hid_dim].
      targets: List of 1D batch-sized int32 Tensors of the same length as inputs.
      weights: List of 1D batch-sized float-Tensors of the same length as inputs.
      loss_function: Sampled softmax function (inputs, labels) -> loss
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      name: Optional name for this operation, defaults to 'sequence_loss'.

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(inputs) is different from len(targets) or len(weights).
    """
    with tf.op_scope(inputs + targets + weights, name, 'sampled_sequence_loss'):
        cost = tf.reduce_sum(sequence_loss_by_example(
            inputs, targets, weights, loss_function,
            average_across_timesteps=average_across_timesteps))
        if average_across_batch:
            batch_size = tf.shape(targets[0])[0]
            return cost / tf.cast(batch_size, tf.float32)
        else:
            return cost


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError('`args` must be specified')
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))
        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            'Bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def sampled_softmax_loss_example(weights,
                                 biases,
                                 inputs,
                                 labels,
                                 num_sampled,
                                 num_classes,
                                 num_true=1,
                                 sampled_values=None,
                                 remove_accidental_hits=True,
                                 partition_strategy="mod",
                                 name="sampled_softmax_loss"):
    """Computes and returns the sampled softmax training loss.

    This is a faster way to train a softmax classifier over a huge number of
    classes.

    This operation is for training only.  It is generally an underestimate of
    the full softmax loss.

    At inference time, you can compute full softmax probabilities with the
    expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

    See our [Candidate Sampling Algorithms Reference]
    (../../extras/candidate_sampling.pdf)

    Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
    ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          [num_classes, dim].  The (possibly-sharded) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          True.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).

    Returns:
      A `batch_size` 1-D tensor of per-example sampled softmax losses.

    """
    logits, labels = tf.nn._compute_sampled_logits(
        weights,
        biases,
        inputs,
        labels,
        num_sampled,
        num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
    sampled_losses = nn_ops.softmax_cross_entropy_with_logits(tf.clip_by_value(logits, 1e-10, 1-1e-10), labels)
    # sampled_losses is a [batch_size] tensor.
    return sampled_losses
