from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Real

from .snn import Solution


def spike_latency_encoding(x: Array, threshold: Real = 0.05, tau: Real = 1.0) -> Array:
    """Encode a continuous signal into a spike train using latency coding.

    **Arguments**:

    - `x`: A continuous signal of shape `(batch, dim)`.
    - `threshold`: The threshold value for spike generation.
    - `tau`: The time constant of the encoding.

    **Returns**:

    - The spike train. A function that takes as input a scalar `t`
        and returns an array of shape `(batch, dim)`.
    """

    def _encoding(_x):
        return tau * jnp.log(_x / (_x - threshold))

    out = jnp.where(x > threshold, _encoding(x), 1e2)
    assert isinstance(out, Array)
    return out


def first_spike_decoding(sol: Solution, out_size: Optional[Int] = None) -> Array:
    if out_size is None:
        out_size = sol.spike_marks.shape[-1]
    st = sol.spike_times
    sm = sol.spike_marks[:, :, -out_size:]

    @partial(jax.vmap, in_axes=2)
    def _decode(_sm):
        out = jnp.where(_sm, st, jnp.inf)
        return jnp.min(out, axis=1)

    return _decode(sm)
