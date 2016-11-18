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
"""Tests for tensorflow.models.ptb_lstm.ptb_reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import google3
import tensorflow as tf

from google3.experimental.users.dzmitryb.lambada import reader


class ReaderTest(tf.test.TestCase):

  def setUp(self):
    self._string_data = "".join(
        ["hello there i am\n",
         "rain as hello there am"])
    self._raw_data = [4, 3, 2, 1, 0,
                      5, 6, 0,
                      1, 1, 0,
                      3, 4, 1, 0,
                      4, 5, 6, 0]

  def testPtbRawData(self):
    tmpdir = tf.test.get_temp_dir()
    for suffix in "train", "valid", "test":
      filename = os.path.join(tmpdir, "ptb.%s.txt" % suffix)
      with tf.gfile.GFile(filename, "w") as fh:
        fh.write(self._string_data)
    output, _, _, _ = reader.ptb_raw_data(tmpdir)
    self.assertEqual(list(output), [1, 2, 5, 0, 3, 6, 4, 1, 2, 0])

  def testReadPaddedSentences(self):
    class FakeVocab(object):
      eos = 0
      bos = 9
    batch_size = 2

    vocab = FakeVocab()
    output = list(reader.add_bos(
      reader.read_padded_sentences(
        self._raw_data, vocab, batch_size, max_length=None),
      vocab))
    self.assertEqual(len(output), 2)
    o1, o2 = (output[0], output[1])
    self.assertEqual(
        o1, ([[9, 4, 3, 2, 1, 0], [9, 5, 6, 0, 0, 0]], [6, 4]))
    self.assertEqual(
        o2, ([[9, 1, 1, 0, 0], [9, 3, 4, 1, 0]], [4, 5]))

    vocab = FakeVocab()
    output = list(reader.add_bos(
      reader.read_padded_sentences(
        self._raw_data, vocab, batch_size,
        max_length=3, truncate=True),
      vocab))
    self.assertEqual(len(output), 2)
    o1, o2 = (output[0], output[1])
    self.assertEqual(
        o1, ([[9, 4, 3, 2], [9, 5, 6, 0]], [4, 4]))
    self.assertEqual(
        o2, ([[9, 1, 1, 0], [9, 3, 4, 1]], [4, 4]))

  def testReadContinuoslyWith(self):
    batch_size = 2
    num_steps = 4
    output = list(reader.read_continuosly(self._raw_data, batch_size, num_steps,
                                           extra_word=True))
    self.assertEqual(len(output), 2)
    o1, o2 = (output[0], output[1])
    self.assertEqual(
        o1[0].tolist(), [[4, 3, 2, 1, 0], [1, 0, 3, 4, 1]])
    self.assertEqual(
        o2[0].tolist(), [[0, 5, 6, 0, 1], [1, 0, 4, 5, 6]])
    self.assertEqual(o1[1], [5, 5])

    output = list(reader.read_continuosly(self._raw_data, batch_size, num_steps))
    self.assertEqual(len(output), 2)
    o1, o2 = (output[0], output[1])
    self.assertEqual(
        o1[0].tolist(), [[4, 3, 2, 1], [1, 0, 3, 4]])
    self.assertEqual(
        o2[0].tolist(), [[0, 5, 6, 0], [1, 0, 4, 5]])
    self.assertEqual(o1[1], [4, 4])


if __name__ == "__main__":
  tf.test.main()
