from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractPath
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, Float, Real


def interleave(arr1: Array, arr2: Array) -> Array:
    out = jnp.empty((arr1.size + arr2.size,), dtype=arr1.dtype)
    out = out.at[0::2].set(arr2)
    out = out.at[1::2].set(arr1)
    return out


def marcus_lift(
    t1: RealScalarLike,
    spike_times: Float[Array, " max_spikes"],
    spike_mask: List[Float[Array, " max_spikes"]],
) -> Float[Array, " 2_max_spikes"]:
    num_neurons = len(spike_mask)
    finite_spikes = jnp.where(jnp.isfinite(spike_times), spike_times, t1).reshape((-1, 1))
    spike_cumsum = jnp.array(jtu.tree_map(lambda x: jnp.cumsum(x), spike_mask), dtype=jnp.float32).T
    spike_cumsum_shift = jnp.roll(spike_cumsum, 1, axis=0)
    spike_cumsum_shift = spike_cumsum_shift.at[0, :].set(jnp.zeros(num_neurons))
    arr1 = jnp.hstack([finite_spikes, spike_cumsum])
    arr2 = jnp.hstack([finite_spikes, spike_cumsum_shift])
    return jax.vmap(interleave, in_axes=1)(arr1, arr2).T


class SpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    num_spikes: int
    spike_times: Array
    spike_cumsum: Array
    num_neurons: int

    def __init__(self, t0, t1, spike_times, spike_mask):
        self.t0 = t0
        self.t1 = t1
        self.num_spikes = spike_times.shape[0]
        self.spike_times = jnp.insert(spike_times, 0, t0)
        self.spike_cumsum = jnp.array(
            jtu.tree_map(lambda x: jnp.cumsum(jnp.insert(x, 0, jnp.array(False))), spike_mask),
            dtype=float,
        )
        self.num_neurons = self.spike_cumsum.shape[0]

    def evaluate(self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        idx = jnp.searchsorted(self.spike_times, t0)
        idx = jnp.where(idx > 0, idx - 1, idx)
        out = jax.lax.dynamic_slice(self.spike_cumsum, (0, idx), (self.num_neurons, 1))[:, 0]
        return out
