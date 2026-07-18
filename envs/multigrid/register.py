# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Register MultiGrid environments with OpenAI gym."""

from envs import registration


def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None, kwargs=None):
  """Register a new environment with OpenAI gym based on id."""
  assert env_id.startswith("MultiGrid-")
  if env_id in registration.registry.env_specs:
    del registration.registry.env_specs[env_id]

  reg_kwargs = dict(
    id=env_id,
    entry_point=entry_point,
    reward_threshold=reward_threshold,
    kwargs=kwargs,
  )

  if max_episode_steps is not None:
    reg_kwargs.update({'max_episode_steps': max_episode_steps})

  registration.register(**reg_kwargs)
