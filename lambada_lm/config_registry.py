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
"""Configuration management."""

class ConfigRegistry(object):

  def __init__(self):
    self._configs = {}

  def __getitem__(self, name):
    # copy on read
    return dict(self._configs[name])

  def register(self, name, config):
    if name in self._configs:
      raise KeyError("Config already registered " + name)
    self._configs[name] = dict(config)

config_registry = ConfigRegistry()
