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
"""Configurations."""

from lambada_lm.config_registry import config_registry
config_registry.register('default', {
  # Data
  'read' : 'continuosly_with_extra_word',
  'eval_read': 'continuosly_with_extra_word',
  'num_steps' : 100,
  'eval_num_steps' : 100,

  # Schedule
  'monitoring_frequency' : 100,
  'saving_frequency' : 5000,
  'max_batches_per_epoch' : 5000,
  'num_epochs' : 100,
  'start_annealing' : 20,
  'lr_decay' : 0.8,

  # Model
  'init_scale' : 0.1,
  'forget_bias' : 0.0,
  'dim' : 128,
  'architecture' : 'lstm',
  'act' : 'relu',
  'width' : -1,


  # Optimization
  'optimizer' : 'GradientDescentOptimizer',
  'batch_size' : 32,
  'learning_rate' : 1.0,
  'lr_min': 0.000001,
  'momentum' : 0.9,
  'epsilon' : 1e-8,
  'max_grad_norm': 5.0,
  'next_worker_delay' : 1500,
})

c = config_registry['default']
c['dim'] = 512
c['read'] = 'shards_continuosly_with_bos'
c['eval_read'] = 'padded_sentences_with_bos'
c['eval_num_steps'] = 210
config_registry.register('lambada', c)

c = config_registry['lambada']
c['optimizer'] = 'AdamOptimizer'
c['learning_rate'] = 0.001
config_registry.register('lambAdam', c)

c = config_registry['lambAdam']
c['architecture'] = 'conv'
c['width'] = 5
config_registry.register('lambAdamConv', c)
