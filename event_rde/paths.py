import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractPath
from jaxtyping import Array, PyTree, Real


class SpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    num_spikes: int
    spike_times: Array
    spike_cumsum: PyTree[Array]

    def __init__(self, t0, t1, spike_times, spike_mask):
        self.t0 = t0
        self.t1 = t1
        self.spike_times = spike_times
        self.num_spikes = spike_times.shape[0]
        self.spike_cumsum = jtu.tree_map(lambda x: jnp.cumsum(x), spike_mask)

    def evaluate(self, t0, t1=None, left=True):
        del left
        assert t0 >= self.t0
        if t1 is not None:
            assert t1 <= self.t1
            return self.evaluate(t1 - t0)
        idx = jnp.searchsorted(self.spike_times, t0)
        out = jtu.tree_map(lambda x: jax.lax.dynamic_slice(x, (idx,), (1,))[0], self.spike_cumsum)
        return out
