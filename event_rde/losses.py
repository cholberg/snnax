import functools as ft

import jax
import jax.numpy as jnp
import signax
from jaxtyping import Array, Float, Real

from .solution import Solution


def expected_signature(y: Float[Array, ""], depth: int) -> Array:
    signatures = jax.vmap(ft.partial(signax.signature, depth=depth))(y)
    return jnp.mean(signatures, axis=0)


def expected_signature_loss(
    y_1: Float[Array, "... dim"], y_2: Float[Array, "... dim"], depth: int
) -> Real:
    spike_counts_1 = jnp.max(y_1[:, :, 1:], axis=1)
    spike_counts_2 = jnp.max(y_2[:, :, 1:], axis=1)
    spike_counts = jnp.minimum(spike_counts_1, spike_counts_2)
    y_1_trunc = y_1.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_1[:, :, 1:], spike_counts))
    y_2_trunc = y_2.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_2[:, :, 1:], spike_counts))
    sig_1 = expected_signature(y_1_trunc, depth)
    sig_2 = expected_signature(y_2_trunc, depth)
    return jnp.mean((sig_1 - sig_2) ** 2)


def coord_spikes(sol: Solution) -> Array:
    spike_times = sol.spike_times
    spike_times = jnp.where(jnp.isinf(spike_times), sol.t1, spike_times)
    spike_marks = sol.spike_marks

    @jax.vmap
    def _outer(st, sm):
        @ft.partial(jax.vmap, in_axes=1)
        def _inner(col):
            out = jnp.full_like(col, sol.t1)
            out = jnp.where(col, st, out)
            return out

        return _inner(sm)

    out = _outer(spike_times, spike_marks)
    assert isinstance(out, Array)
    return out


def first_spike_loss(sol_1: Solution, sol_2: Solution) -> Real:
    coord_spikes_1 = coord_spikes(sol_1)
    coord_spikes_2 = coord_spikes(sol_2)
    return jnp.mean((coord_spikes_1 - coord_spikes_2)[:, :, 0] ** 2)


def spike_MAE_loss(sol_1: Solution, sol_2: Solution) -> Real:
    coord_spikes_1 = coord_spikes(sol_1)
    coord_spikes_2 = coord_spikes(sol_2)
    out = jnp.mean(jnp.abs(coord_spikes_1 - coord_spikes_2))
    return out


def spike_MSE_loss(sol_1: Solution, sol_2: Solution) -> Real:
    coord_spikes_1 = coord_spikes(sol_1)
    coord_spikes_2 = coord_spikes(sol_2)
    out = jnp.mean((coord_spikes_1 - coord_spikes_2) ** 2)
    return out
