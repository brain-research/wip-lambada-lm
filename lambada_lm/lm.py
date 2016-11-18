# Copyright 2016 Google Inc.
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
"""Train language models, autoencoders, VAEs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pprint
import math
import tempfile
from collections import defaultdict

import numpy
import tensorflow as tf
from lambada_lm import reader

from lambada_lm.config_registry import config_registry
import lambada_lm.configs

flags = tf.flags
logging = tf.logging

# Distributed or local training
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps tasks used.')
flags.DEFINE_integer('task', 0,
                     'Task id of the replica running the training.')

# Mode and configuration to run
flags.DEFINE_string(
    "mode", "train",
    "Available modes: train, sample, greedy")
flags.DEFINE_string(
    "config", "default",
    "Configuration")

# Pathes
flags.DEFINE_string("dataset", None, "Dataset to train on")
flags.DEFINE_string("data_path", None,
                    "Path to the dataset")
flags.DEFINE_string("train_path", None,
                    "Path to save the model and summaries")

# Evaluation
flags.DEFINE_integer("num_examples", -1, "Number of examples to consider")
flags.DEFINE_string("output", None, "Destination for evaluation results")
flags.DEFINE_string("part", "test", "Path to use for evaluation")

# Make all configuration options available as flags
config_option_help = "sets the respective configuration option"
for option, value in config_registry['default'].items():
  if isinstance(value, bool):
    flags.DEFINE_bool(option, None, config_option_help)
  elif isinstance(value, str):
    flags.DEFINE_string(option, None, config_option_help)
  elif isinstance(value, int):
    flags.DEFINE_integer(option, None, config_option_help)
  elif isinstance(value, float):
    flags.DEFINE_float(option, None, config_option_help)
  elif isinstance(value, list):
    flags.DEFINE_list(option, None, config_option_help)
  else:
    raise ValueError("unknown option type " + str(type(value)))

FLAGS = flags.FLAGS


def affine(inputs, scope, num_outputs=None, w=None):
  num_inputs = inputs.shape[-1]
  if not num_outputs:
    num_outputs = num_inputs
  with tf.variable_scope(scope):
    if w is None:
      w = tf.get_variable(shape=(num_inputs, num_outputs), name='W')
    b = tf.get_variable(shape=(num_outputs,), name='b')
    return tf.matmul(inputs, w) + b


def activation(func, inputs):
  if func == 'relu':
    return tf.nn.relu(inputs)
  elif func == 'tanh':
    return tf.tanh(inputs)
  raise ValueError("unknown activation function: {}".format(func))


def _add_summary_value(summary, tag, simple_value):
  value = summary.value.add()
  value.tag = tag
  value.simple_value = simple_value


class LanguageModel(object):
  """Language model graph builder.

  This is just light-weight functor, not a full-fledged incapsulation.

  """
  def __init__(self, config):
    self.config = config
    c = config


  def _embed(self, inputs):
    c = self.config


  def _build_graph(self):
    c = self.config

    inputs = tf.placeholder(tf.int32, [c['batch_size'], None],
                            name='inputs')
    input_lengths = tf.placeholder(tf.int32, [c['batch_size']],
                            name='input_lengths')
    num_steps = tf.shape(inputs)[1] - 1

    embeddings = tf.get_variable("embeddings", [c['vocab_size'], c['dim']])
    embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs[:, :-1])
    targets = inputs[:, 1:]

    if c['architecture'] == 'lstm':
      cell = tf.nn.rnn_cell.BasicLSTMCell(c['dim'], forget_bias=0.0)
      (outputs, state) = tf.nn.dynamic_rnn(cell, embedded_inputs,
                                          dtype=tf.float32)
    elif c['architecture'] == 'conv':
      padded_inputs = tf.pad(embedded_inputs,
                             [[0, 0], [c['width'] - 1, 0], [0, 0]])
      filter_ = tf.get_variable("filter_", [c['width'], c['dim'], c['dim']])
      convolved = tf.nn.conv1d(padded_inputs, filter_, 1, "VALID")
      outputs = activation(c['act'], convolved)
    else:
      raise ValueError("Unknown architecture {}".format(c['architecture']))

    softmax_w = tf.get_variable("softmax_w", [c['dim'], c['vocab_size']])
    softmax_b = tf.get_variable("softmax_b", [c['vocab_size']])
    outputs = tf.reshape(outputs, (-1, c['dim']))
    logits = tf.nn.log_softmax(tf.matmul(outputs, softmax_w) + softmax_b)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
    losses = cross_entropy(logits, tf.reshape(targets, (-1,)))
    losses = tf.reshape(losses, tf.shape(targets))
    predictions = tf.to_int32(tf.reshape(tf.arg_max(logits, 1), tf.shape(targets)))
    logits = tf.reshape(logits, (c['batch_size'], num_steps, -1))
    correct = tf.to_float(tf.equal(predictions, targets))

    weights = tf.to_float(tf.less(tf.range(num_steps)[None, :],
                                  input_lengths[:, None] - 1))
    losses *= weights
    correct *= weights
    neg_loglikelihood = tf.reduce_sum(losses, 1)
    num_correct = tf.reduce_sum(correct, 1)

    graph =  {'inputs': inputs,
              'input_lengths': input_lengths,
              'losses': losses,
              'logits': logits,
              'neg_loglikelihood': neg_loglikelihood,
              'num_correct': num_correct}
    return graph

  def build_graph(self):
    initializer = tf.random_uniform_initializer(-self.config['init_scale'],
                                                self.config['init_scale'])
    with tf.variable_scope('lm', initializer=initializer):
      return self._build_graph()


class Monitor(object):

  def __init__(self, summary_writer, summary_prefix):
    self._summary_writer = summary_writer
    self._summary_prefix = summary_prefix
    self._reset()

  def _reset(self):
    self.start_time = time.time()
    self.total_neg_loglikelihood = 0.
    self.total_correct = 0
    self.steps = 0
    self.sentences = 0
    self.words = 0
    self.words_including_padding = 0
    self.observed = defaultdict(float)

  def _add_summary_value(self, summary, tag, simple_value):
    _add_summary_value(summary, '_'.join([self._summary_prefix, tag]), simple_value)

  def monitor(self):
    perplexity_so_far = numpy.exp(self.total_neg_loglikelihood / self.words)
    accuracy = self.total_correct / self.words
    neg_loglikelihood_so_far = self.total_neg_loglikelihood / self.sentences
    time_taken = time.time() - self.start_time
    for key in self.observed:
      self.observed[key] /= self.sentences

    print("STATUS")
    print("steps done: {}, ll: {:.3f}, ppl: {:.3f}, acc: {:.3f}, "
          "speed: {:.0f} wps, time_per_step: {:.03f} s, efficiency: {:.2f}".format(
              self.step_number,
              -neg_loglikelihood_so_far,
              perplexity_so_far,
              accuracy,
              self.words / time_taken, time_taken / self.steps,
              self.words / float(self.words_including_padding)))
    print("OBSERVED VARIABLES")
    for key, value in self.observed.items():
      print(key + ":", value)

    if self._summary_writer:
        summary = tf.Summary()
        self._add_summary_value(summary, 'nll', neg_loglikelihood_so_far)
        self._add_summary_value(summary, 'perplexity', perplexity_so_far)
        self._add_summary_value(summary, 'accuracy', accuracy)
        self._add_summary_value(summary, 'learning_rate', self.learning_rate)
        for key, value in self.observed.items():
          self._add_summary_value(summary, key, value)
        self._summary_writer.add_summary(summary, self.step_number)

    self._reset()

    return neg_loglikelihood_so_far, perplexity_so_far


def run_epoch(session, config, graph, iterator, ops=None,
              summary_writer=None, summary_prefix=None, saver=None):
  """Runs the model on the given data."""
  if not ops:
    ops = []

  def should_monitor(step):
    return step and c['monitoring_frequency'] and (step + 1) % c['monitoring_frequency'] == 0
  def should_save(step):
    return step and c['saving_frequency'] and (step + 1) % c['saving_frequency'] == 0

  # Shortcuts, ugly but still increase readability
  c = config
  g = graph
  m = Monitor(summary_writer, summary_prefix)

  while g['step_number'].eval() < FLAGS.task * c['next_worker_delay']:
    pass

  # Statistics
  for step, (inputs, lengths) in enumerate(iterator):
    # Define what we feed
    feed_dict = {g['inputs']: inputs,
                 g['input_lengths']: lengths}

    # Define what we fetch
    fetch = dict(g['observed'])
    fetch['total_neg_loglikelihood'] = g['total_neg_loglikelihood']
    fetch['total_correct'] = g['total_correct']
    fetch['_ops'] = ops

    # RUN!!!
    r = session.run(fetch, feed_dict)

    # Update the monitor accumulators
    m.total_neg_loglikelihood += r['total_neg_loglikelihood']
    m.total_correct += r['total_correct']
    # We do not predict the first words, that's why
    # batch_size has to subtracted from the total
    m.steps += 1
    m.words += sum(lengths) - c['batch_size']
    m.sentences += c['batch_size']
    m.words_including_padding += c['batch_size'] * len(inputs[0])
    m.step_number = g['step_number'].eval()
    m.learning_rate = float(g['learning_rate'].eval())
    for key in g['observed']:
      m.observed[key] += r[key]

    if should_monitor(step):
      tf.logging.info('monitor')
      result = m.monitor()
    if saver and should_save(step):
      print("saved")
      saver.save(session, os.path.join(FLAGS.train_path, 'model'))

  if not should_monitor(step):
    result = m.monitor()
  if saver:
    saver.save(session, os.path.join(FLAGS.train_path, 'model'))
  return result


def get_config(**changes):
  config = config_registry[FLAGS.config]
  config.update(changes)
  return config


def cmd_config_changes():
  return {option: getattr(FLAGS, option)
          for option in config_registry['default']
          if getattr(FLAGS, option) is not None}


def train(config, data_parts, vocab):
  valid_config = dict(config)
  valid_config['monitoring_frequency'] = 0
  test_config = dict(config)
  test_config['monitoring_frequency'] = 0
  test_config['batch_size'] = 1

  train_reader, valid_reader, test_reader = [
    reader.Reader(
      c['eval_read'] if is_eval else c['read'],
      data_parts[part], vocab,
      c['batch_size'],
      c['eval_num_steps'] if is_eval else c['num_steps'],
      None if is_eval else c['max_batches_per_epoch'])
    for part, c, is_eval in zip(['train', 'valid', 'test'],
                                [config, valid_config, test_config],
                                [False, True, True])]

  if FLAGS.ps_tasks:
    tf.get_variable_scope().set_caching_device('/job:worker')
  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # All the variables should be created here!
    graph = LanguageModel(config).build_graph()
    valid_graph = LanguageModel(valid_config).build_graph()
    test_graph = LanguageModel(test_config).build_graph()

    learning_rate = tf.Variable(0.0, trainable=False, name='learning_rate')
    epoch_number = tf.Variable(0, trainable=False, name='epoch_number')
    step_number = tf.Variable(0, trainable=False, name='step_number')

    actual_objective = tf.reduce_mean(graph['neg_loglikelihood'])

    trainable_variables = tf.trainable_variables()
    gradients = tf.gradients(actual_objective, trainable_variables)
    gradients = [tf.check_numerics(grad, 'nan in gradients')
                 for grad in gradients]
    gradient_norm = tf.global_norm(gradients)
    grads, _ = tf.clip_by_global_norm(
        gradients, config['max_grad_norm'], use_norm=gradient_norm)
    optimizer_args = {'learning_rate' : learning_rate}
    if c['optimizer'] == 'AdamOptimizer':
      optimizer_args['beta1'] = c['momentum']
      optimizer_args['epsilon'] = c['epsilon']
    optimizer = getattr(tf.train, c['optimizer'])(**optimizer_args)
    train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    learning_rate_update = learning_rate.assign(
      tf.maximum(
        config['learning_rate'] *
        config['lr_decay'] **
        tf.to_float(tf.maximum(epoch_number - config['start_annealing'], 0)),
        config['lr_min']))
    step_number_update = step_number.assign(step_number + 1)
    epoch_number_update = epoch_number.assign(epoch_number + 1)

    saver = tf.train.Saver()

  for g in [graph, valid_graph, test_graph]:
    g['step_number'] = step_number
    g['learning_rate'] = learning_rate
    # These trivial computation have to be declared here,
    # because the graph can not be modified in run_epoch
    g['total_neg_loglikelihood'] = tf.reduce_sum(g['neg_loglikelihood'])
    g['total_correct'] = tf.reduce_sum(g['num_correct'])
    observed = {}
    if g == graph:
      # Gradient norm is only monitored during training
      # It has to be multiplied by `batch_size`, because
      # it will be later treated as a sum over all examples
      observed['gradient_norm'] = gradient_norm * c['batch_size']
    g['observed'] = observed

  summary_writer = tf.train.SummaryWriter(FLAGS.train_path, flush_secs=10.)
  if FLAGS.task == 0:
    print("I am chief!")
    with tf.gfile.GFile('graph.pbtxt', 'w') as dest:
      print(tf.get_default_graph().as_graph_def(), file=dest)
    summary_writer.add_graph(tf.get_default_graph())

  # Main loop with automatic recovery
  stopped = False
  while not stopped:
    try:
      supervisor = tf.train.Supervisor(
        logdir=FLAGS.train_path,
        is_chief=FLAGS.task == 0, global_step=step_number,
        saver=saver, save_model_secs=0,
        summary_op=None,
        summary_writer=summary_writer if FLAGS.task == 0 else None)
      with supervisor.managed_session() as session:
        with session.as_default():
          while True:
            i = epoch_number.eval()
            if i > c['num_epochs']:
              stopped = True
              break

            print("Epoch: %d Learning rate: %.6f" %
                  (i + 1, session.run(learning_rate_update)))
            _, train_perplexity = run_epoch(
              session, config, graph,
              train_reader.read_next_epoch(),
              [train_op, step_number_update],
              summary_writer if supervisor.is_chief else None, 'train',
              saver if supervisor.is_chief else None)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            if supervisor.is_chief:
              _, valid_perplexity = run_epoch(
                session, valid_config, valid_graph,
                valid_reader.read_next_epoch(),
                summary_writer=summary_writer, summary_prefix='valid')
              print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            session.run(epoch_number_update)

          if supervisor.is_chief:
            _, test_perplexity = run_epoch(
              session, test_config, test_graph,
              test_reader.read_next_epoch(),
              summary_writer=summary_writer, summary_prefix='test')
            print("Test Perplexity: %.3f" % test_perplexity)
    except tf.errors.AbortedError:
      pass
    except tf.errors.InvalidArgumentError as e:
      if 'nan in gradients' in e.message:
        tf.logging.error('nan in gradients!')
        if FLAGS.task > 0:
          time.sleep(1.0)
      else:
        raise


def evaluate(config, data_parts, vocab):
  c = config
  c['batch_size'] = 1

  if FLAGS.part == 'stdin':
    reader_ = reader.Reader(
        c['eval_read'],
        reader.Words(sys.stdin, vocab), vocab,
        1, c['eval_num_steps'], None)
  elif FLAGS.part == 'train':
    reader_ = reader.Reader(
      c['read'],
      data_parts['train'], vocab,
      1, c['num_steps'], None)
  else:
    reader_ = reader.Reader(
      c['eval_read'],
      data_parts[FLAGS.part], vocab,
      1, c['eval_num_steps'], None)

  g  = LanguageModel(config).build_graph()

  num_examples = 0.
  num_words = 0.
  total_neg_loglikelihood = 0.
  total_last_neg_loglikelihood = 0.
  total_correct = 0.
  total_last_correct = 0.
  last_correct_ranks = []

  dest = sys.stdout
  if FLAGS.output:
    dest = tf.gfile.GFile(FLAGS.output, 'w')

  with tf.Session() as session:
    saver = tf.train.Saver()
    logging.info("Restoring...")
    saver.restore(session, os.path.join(FLAGS.train_path, 'model'))
    logging.info("Ready")

    for step, (inputs, lengths) in enumerate(reader_.read_next_epoch()):
      if FLAGS.num_examples >= 0 and step == FLAGS.num_examples:
        break

      input_ = inputs[0]
      num_words += len(input_) - 1

      r = session.run(
        {'losses' : g['losses'],
         'logits' : g['logits']},
        feed_dict={g['inputs']: inputs,
                   g['input_lengths']: lengths})
      for key in r:
        r[key] = r[key][0]

      num_examples += 1
      total_neg_loglikelihood += r['losses'].sum()

      print("INPUT {}: {}".format(step + 1, vocab.decode(input_)), file=dest)
      for i in range(1, len(input_)):
        loss = r['losses'][i - 1]
        logits = r['logits'][i - 1]
        ranking = numpy.argsort(-logits)
        is_correct = int(ranking[0] == input_[i])
        total_correct += is_correct
        if i + 2 == len(input_):
          # The second last token should be considered,
          # because the last one is <eos>
          total_last_correct += is_correct
          total_last_neg_loglikelihood += loss
          last_correct_ranks.append(numpy.where(ranking == input_[i])[0][0] + 1)
        top3_str = " ".join(["({}, {:.3f})".format(vocab.id_to_word(index), logits[index])
                             for index in ranking[:3]])
        print("{:20}{:.3f}  {}".format(vocab.id_to_word(input_[i]), loss, top3_str), file=dest)

      if step % 100 == 0:
        dest.flush()

  print("AVERAGE NEG_LOGIKELIHOOD PER SENTENCE: {}".format(
    total_neg_loglikelihood / num_examples), file=dest)
  print("AVERAGE ACCURACY: {}".format(
    total_correct / num_words), file=dest)
  print("AVERAGE LAST WORD ACCURACY: {}".format(
    total_last_correct / num_examples), file=dest)
  print("AVERAGE LAST WORD PERPLEXITY: {}".format(
    numpy.exp(total_last_neg_loglikelihood / num_examples)), file=dest)
  print("MEDIAN LAST CORRECT WORD RANK: {}".format(
    numpy.median(last_correct_ranks)), file=dest)

  dest.flush()
  if FLAGS.output:
    dest.close()


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data directory")

  tf.get_variable_scope()._reuse = None

  data_parts, vocab = reader.get_dataset_raw_data(FLAGS.dataset, FLAGS.data_path)

  cmd_changes = {}
  cmd_changes['vocab_size'] = vocab.size
  cmd_changes['eos'] = vocab.eos
  cmd_changes['bos'] = vocab.bos
  cmd_changes.update(cmd_config_changes())
  config = get_config(**cmd_changes)

  print("CONFIGURATION")
  pprint.pprint(config)
  with open('config.txt', 'w') as dst:
    pprint.pprint(config, stream=dst)

  if FLAGS.mode == 'train':
    train(config, data_parts, vocab)
  elif FLAGS.mode == 'eval':
    evaluate(config, data_parts, vocab)
  else:
    raise ValueError("unknown mode")


if __name__ == "__main__":
  tf.app.run()
