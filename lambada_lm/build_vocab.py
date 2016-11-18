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
"""Builds a vocabulary for a given text file. """

import tensorflow as tf

from lambada import reader

flags = tf.flags
flags.DEFINE_string("text", None, "Text file to process")
flags.DEFINE_integer("top_k", None, "Top k words to keep")
flags.DEFINE_string("vocab", None, "Destination for the created vocabulary")

FLAGS = flags.FLAGS


def main(unused_argv):
  vocab = reader.Vocabulary.build(FLAGS.text, FLAGS.top_k)
  vocab.save(FLAGS.vocab)


if __name__ == '__main__':
  tf.app.run()
