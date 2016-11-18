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
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy
import tensorflow as tf


class Vocabulary(object):
  """Class that holds a vocabulary for the dataset.

  Note: copied from learning/brain/models/lm_1b

  """
  BOS = '<bos>'
  EOS = '<eos>'
  UNK = '<unk>'
  SHIFT = '<shift>'
  REDUCE = '<reduce>'

  def __init__(self, filename_or_words):
    if isinstance(filename_or_words, str):
      with tf.gfile.Open(filename_or_words) as f:
        words = [line.strip() for line in f]
    else:
      words = list(filename_or_words)

    self._id_to_word = []
    self._word_to_id = {}
    self._unk = -1
    self._bos = -1
    self._eos = -1

    for idx, word_name in enumerate(words):
      if word_name == Vocabulary.BOS:
        self.bos = idx
      elif word_name == Vocabulary.EOS:
        self.eos = idx
      elif word_name == Vocabulary.UNK:
        self.unk = idx
      elif word_name == Vocabulary.SHIFT:
        self.shift = idx
      elif word_name == Vocabulary.REDUCE:
        self.reduce = idx

      self._id_to_word.append(word_name)
      self._word_to_id[word_name] = idx

  @property
  def size(self):
    return len(self._id_to_word)

  def word_to_id(self, word):
    if word in self._word_to_id:
      return self._word_to_id[word]
    return self.unk

  def id_to_word(self, cur_id):
    return self._id_to_word[cur_id]

  def decode(self, cur_ids):
    return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

  def encode(self, sentence):
    word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
    return numpy.array([self.bos] + word_ids + [self.eos],
                       dtype=numpy.int32)

  @staticmethod
  def build(filename, top_k=None):
    data = Words(filename, None)

    counter = collections.Counter(data)
    # It was not immediately clear to me
    # if counter.most_common() selects consistenly among
    # the words with the same counts. Hence, let's just sort.
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words = [w for w, _ in count_pairs]
    if top_k:
      words = words[:top_k]
    for special in [Vocabulary.BOS, Vocabulary.EOS, Vocabulary.UNK]:
      if not special in words:
        words.append(special)

    return Vocabulary(words)

  def save(self, filename):
    with tf.gfile.Open(filename, 'w') as f:
      for word in self._id_to_word:
        print(word, file=f)


class Words(object):

  def __init__(self, filename_or_file, vocab):
    self._filename_or_file = filename_or_file
    self._vocab = vocab

  def _iter_impl(self, f):
      for line in f:
        for word in line.replace("\n", " " + Vocabulary.EOS + " ").split():
          if self._vocab:
            yield self._vocab.word_to_id(word)
          else:
            yield word

  def __iter__(self):
    if isinstance(self._filename_or_file, str):
      with tf.gfile.GFile(self._filename_or_file, "r") as f:
        return self._iter_impl(f)
    else:
      return self._iter_impl(self._filename_or_file)


class ShardedWords(object):

  def __init__(self, filename_pattern, vocab):
    self._shards = tf.gfile.Glob(filename_pattern)
    self._vocab = vocab

  @property
  def num_shards(self):
    return len(self._shards)

  def __iter__(self):
    shard_order = numpy.random.choice(self._shards, len(self._shards), replace=False)
    for shard in shard_order:
      for word in Words(shard, self._vocab):
        yield word

  def deterministic_iter(self, start_shard):
    n = len(self._shards)
    shard_order = (numpy.arange(n) + start_shard) % n
    for shard_number in shard_order:
      for word in Words(self._shards[shard_number], self._vocab):
        yield word


### DATASETS


def get_dataset_raw_data(dataset, data_path):
  if dataset == 'ptb':
    return ptb_raw_data(data_path)
  elif dataset == 'books':
    return books_sharded_raw_data(data_path)
  elif dataset == 'lambada':
    return lambada_sharded_raw_data(data_path)
  elif dataset == 'standard':
    return standard_raw_data(data_path)
  else:
    raise ValueError("Unknown dataset")


def ptb_raw_data(data_path):
  # Returns lists of words ids
  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  vocab = Vocabulary.build(train_path)
  train_data = Words(train_path, vocab)
  valid_data = Words(valid_path, vocab)
  test_data = Words(test_path, vocab)
  return ({'train' : train_data,
           'valid' : valid_data,
           'test' : test_data},
          vocab)


def books_raw_data(data_path):
  # Returns iterators over all words
  train_path = os.path.join(data_path, "books_train.txt")
  valid_path = os.path.join(data_path, "books_valid.txt")
  test_path = os.path.join(data_path, "books_test.txt")
  vocab_path = os.path.join(data_path, "books_vocab.txt")

  vocab = Vocabulary(vocab_path)
  train_data = Words(train_path, vocab)
  valid_data = Words(valid_path, vocab)
  test_data = Words(test_path, vocab)
  return ({'train': train_data,
           'valid': valid_data,
           'test': test_data},
          vocab )


def books_sharded_raw_data(data_path):
  # Returns iterators over all words
  train_path = os.path.join(data_path, "books_train_shard.*")
  valid_path = os.path.join(data_path, "books_valid.txt")
  test_path = os.path.join(data_path, "books_test.txt")
  vocab_path = os.path.join(data_path, "books_vocab.txt")

  vocab = Vocabulary(vocab_path)
  train_data = ShardedWords(train_path, vocab)
  valid_data = Words(valid_path, vocab)
  test_data = Words(test_path, vocab)
  return ({'train': train_data,
           'valid': valid_data,
           'test': test_data},
          vocab)


def lambada_sharded_raw_data(data_path):
  vocab = Vocabulary(os.path.join(data_path, 'vocab.txt'))
  train_data = ShardedWords(os.path.join(data_path, 'train_shard.*.txt'), vocab)
  valid_data = Words(os.path.join(data_path, 'lambada_development_plain_text.txt'), vocab)
  test_data = Words(os.path.join(data_path, 'lambada_test_plain_text.txt'), vocab)
  control_data = Words(os.path.join(data_path, 'lambada_control_test_data_plain_text.txt'), vocab)
  return ({'train': train_data,
           'valid': valid_data,
           'test': test_data,
           'control': control_data},
          vocab)


def standard_raw_data(data_path):
  vocab = Vocabulary(os.path.join(data_path, 'vocab.txt'))
  train_data = Words(os.path.join(data_path, 'train.txt'), vocab)
  valid_data = Words(os.path.join(data_path, 'valid.txt'), vocab)
  test_data = Words(os.path.join(data_path, 'test.txt'), vocab)
  return ({'train': train_data,
           'valid': valid_data,
           'test': test_data},
          vocab)


### READING METHODS


def _add_padding(batch, padding):
  lengths = [len(s) for s in batch]
  longest = max(len(s) for s in batch)
  return [s + [padding] * (longest - len(s)) for s in batch], lengths


def add_bos(iter_, vocab):
  while True:
    inputs, lengths = next(iter_)
    for i in range(len(inputs)):
      inputs[i] = [vocab.bos] + inputs[i]
      lengths[i] += 1
    yield inputs, lengths


def remove_eos(iter_, vocab):
  while True:
    inputs, lengths = next(iter_)
    max_length = len(inputs[0])
    for i in range(len(inputs)):
      eos_count = inputs[i].count(vocab.eos)
      inputs[i] = ([token for token in inputs[i] if token != vocab.eos]
                   + [0] * eos_count)
      lengths[i] -= eos_count
    yield inputs, lengths


def read_padded_sentences(data, vocab, batch_size, max_length,
                          truncate=False):
  """Reads sentences add pads them.

  Skips sentences with more than `max_length` words,
  *including* EOS. Can also prepends BOS, and as a result
  can return sequences with up to `max_length + 1` elements.

  """
  batch = []
  new_sentence = []
  sentence = list(new_sentence)
  for word in data:
    sentence.append(word)
    if word == vocab.eos:
      if max_length is None or len(sentence) <= max_length:
        batch.append(sentence)
      elif truncate:
        batch.append(sentence[:max_length])
      sentence = list(new_sentence)
    if len(batch) == batch_size:
      yield _add_padding(batch, vocab.eos)
      batch = []

  # Don't yield the incomplete batch


def read_continuosly(data, batch_size, num_steps,
                      extra_word=False):
  # This effectively reads the whole input file
  data = numpy.array(list(data), dtype=numpy.int32)

  data_len = len(data)
  chunk_len = data_len // batch_size
  chunks = numpy.zeros([batch_size, chunk_len], dtype=numpy.int32)
  for i in range(batch_size):
    chunks[i] = data[chunk_len * i:chunk_len * (i + 1)]

  epoch_size = (chunk_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    yield (map(list, chunks[:, i*num_steps:(i+1)*num_steps + int(extra_word)]),
           [num_steps + int(extra_word),] * batch_size)


def read_shards_continuosly(data, batch_size, num_steps):
  if not isinstance(data, ShardedWords):
    tf.logging.warning("not sharded data provided")
    for batch in read_continuosly(data, batch_size, num_steps):
      yield batch
    return
  if batch_size > data.num_shards:
    raise ValueError
  iters = [data.deterministic_iter(i)
           for i in range(batch_size)]
  while True:
    chunks = [[next(it) for i in range(num_steps)]
              for it in iters]
    yield chunks, [num_steps,] * batch_size


## READER


class Reader(object):

  def __init__(self, mode, words, vocab, batch_size, max_length,
               max_batches_per_epoch):
    self.mode = mode
    self.words = words
    self.vocab = vocab
    self.batch_size = batch_size
    self.max_length = max_length
    self.max_batches_per_epoch = max_batches_per_epoch

    self._iterator = None

  def _get_iterator(self):
    if self.mode == 'continuosly_with_extra_word':
      return remove_eos(read_continuosly(
        self.words, self.batch_size, self.max_length,
        extra_word=True), self.vocab)
    elif self.mode == 'shards_continuosly_with_bos':
      return remove_eos(add_bos(
        read_shards_continuosly(
          self.words, self.batch_size, self.max_length),
        self.vocab), self.vocab)
    elif self.mode == 'padded_sentences_with_bos':
      return add_bos(
        read_padded_sentences(
          self.words, self.vocab, self.batch_size, self.max_length),
        self.vocab)
    else:
      raise ValueError

  def read_next_epoch(self):
    def _read():
      steps_done = 0
      while (self.max_batches_per_epoch is None
             or steps_done < self.max_batches_per_epoch):
        try:
          yield next(self._iterator)
          steps_done += 1
        except StopIteration:
          self._iterator = None
          break

    if not self._iterator:
      self._iterator = self._get_iterator()
    return _read()
