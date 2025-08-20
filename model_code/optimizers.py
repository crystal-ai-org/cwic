# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Gradient clipping transformations.

Note that complex numbers are also supported, see
https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
"""

from functools import partial
from typing import NamedTuple
import chex
import jax
import jax.numpy as jnp
from optax import Params, tree_utils as otu
from optax._src import base
from optax._src import linear_algebra
from optax._src import numerics


class ClipByQuantileState(NamedTuple):
    norm_history: chex.Array
    ind: chex.Array


def init_clip_by_global_norm_quantile_state(
    params: Params, history_length: int = 100, top_k: int = 10
) -> ClipByQuantileState:
    del params

    return ClipByQuantileState(
        norm_history=jnp.full((history_length,), jnp.finfo(jnp.float32).max),
        ind=jnp.array(0, dtype=jnp.uint32),
    )


def clip_by_global_norm_quantile(
    history_length: int = 100,
    top_k: int = 10,
) -> base.GradientTransformation:

    def update_fn(updates, state: ClipByQuantileState, params=None):
        del params

        norm = linear_algebra.global_norm(updates)

        thresh = -jnp.partition(-state.norm_history, top_k)[top_k].astype(norm.dtype)
        trigger = norm > thresh

        def clip_fn(x):
            return jax.lax.select(trigger, (x * thresh / (norm + 1e-12)).astype(x.dtype), x)

        updates = jax.tree.map(clip_fn, updates)

        new_state = ClipByQuantileState(
            norm_history=state.norm_history.at[state.ind].set(norm),
            ind=(state.ind + 1) % history_length,
        )

        return updates, new_state

    return base.GradientTransformation(
        partial(
            init_clip_by_global_norm_quantile_state, history_length=history_length, top_k=top_k
        ),
        update_fn,
    )  # type: ignore


class ClipMagnitudeState(NamedTuple):
    pass


def init_clip_magnitude_state(params: Params):
    del params
    return ClipMagnitudeState()


def clip_magnitude(magnitude=10.0) -> base.GradientTransformation:

    def update_fn(updates, state, params=None):
        del params

        def clip_fn(t):
            return jnp.clip(t, min=-magnitude, max=magnitude)

        updates = jax.tree.map(clip_fn, updates)
        return updates, ClipMagnitudeState()

    return base.GradientTransformation(partial(init_clip_magnitude_state), update_fn)  # type: ignore
