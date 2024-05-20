import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import sigkerax
import sigkerax.sigkernel
import signax
from jaxtyping import Array, Float, Real


def expected_signature(y: Float[Array, ""], depth: int) -> Array:
    signatures = jax.vmap(ft.partial(signax.signature, depth=depth))(y)
    return jnp.mean(signatures, axis=0)


@jax.vmap
def cap_spike_times(st, sc, sc_min):
    st_inf = jnp.where(sc <= sc_min, st, -jnp.inf)
    assert isinstance(st_inf, Array)
    st_max = jnp.max(st_inf)
    return jnp.where(jnp.isfinite(st), st, st_max)


@eqx.filter_jit
def expected_signature_loss(
    y_1: Float[Array, "... dim"],
    y_2: Float[Array, "... dim"],
    depth: int,
    match_spikes: bool = True,
) -> Real:
    """Compute the signature kernel MMD between two batches sets of spike trains.

    **Arguments**:

    - `y_1`: A batch of spike trains of shape `[..., dim]`.
    - `y_2`: Another batch of spike trains of shape `[..., dim]`.
    - `depth`: The truncation depth of the signature kernel.
    - `match_spikes`: Whether to match the number of spikes in the two batches.

    **Returns**:

    - A real number representing the signature kernel MMD between the two batches.
    """
    if match_spikes:
        spike_counts_1 = jnp.max(y_1[:, :, 1:], axis=1)
        spike_counts_2 = jnp.max(y_2[:, :, 1:], axis=1)
        spike_counts = jnp.minimum(spike_counts_1, spike_counts_2)
        y_1 = y_1.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_1[:, :, 1:], spike_counts))
        y_2 = y_2.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_2[:, :, 1:], spike_counts))

    sig_1 = expected_signature(y_1, depth)
    sig_2 = expected_signature(y_2, depth)
    return jnp.mean((sig_1 - sig_2) ** 2)


@eqx.filter_jit
def signature_mmd(
    y_1: Float[Array, "... dim"],
    y_2: Float[Array, "... dim"],
    match_spikes: bool = True,
    scales: Optional[Array] = None,
    refinement_factor: int = 1,
) -> Real:
    dim = y_1.shape[-1]
    assert y_2.shape[-1] == dim
    if match_spikes:
        spike_counts_1 = jnp.max(y_1[:, :, 1:], axis=1)
        spike_counts_2 = jnp.max(y_2[:, :, 1:], axis=1)
        spike_counts = jnp.minimum(spike_counts_1, spike_counts_2)
        y_1 = y_1.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_1[:, :, 1:], spike_counts))
        y_2 = y_2.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_2[:, :, 1:], spike_counts))
    if scales is None:
        scales = jnp.ones((dim,))
    sig_kernel = sigkerax.sigkernel.SigKernel(
        scales=scales, refinement_factor=refinement_factor, static_kernel_kind="rbf"
    )
    k_11 = sig_kernel.kernel_matrix(y_1, y_1)
    k_12 = sig_kernel.kernel_matrix(y_1, y_2)
    k_22 = sig_kernel.kernel_matrix(y_2, y_2)
    return jnp.mean(k_11) + jnp.mean(k_22) - 2 * jnp.mean(k_12)


def get_n_first_spikes(
    y: Float[Array, "samples double_spikes _neurons"], n: int
) -> Float[Array, "samples neurons"]:
    @jax.vmap
    def _outer(_y):
        @jax.vmap
        def _inner(k):
            idx = jnp.sum(_y[:, 1:] < k, axis=0)
            return _y[idx, 0]

        return _inner(jnp.arange(n) + 1)

    out = _outer(y)
    return out


@eqx.filter_jit
def spike_MAE_loss(y_1: Float[Array, "... dim"], y_2: Float[Array, "... dim"], n: int) -> Real:
    """Compute the mean absolute error between the average $n$ first spike times
    of two batches of spike trains.

    **Arguments**:

    - `y_1`: A batch of spike trains of shape `[..., dim]`.
    - `y_2`: Another batch of spike trains of shape `[..., dim]`.
    - `n`: The number of first spikes to consider.

    **Returns**:

    - A real number representing the mean absolute error between the average $n$ first spike times.
    """
    first_spikes = get_n_first_spikes(y_1, n)
    avg_first_spikes_1 = jnp.mean(first_spikes, axis=0)
    first_spikes = get_n_first_spikes(y_2, n)
    avg_first_spikes_2 = jnp.mean(first_spikes, axis=0)
    return jnp.mean(jnp.abs(avg_first_spikes_1 - avg_first_spikes_2))


@eqx.filter_jit
def spike_MSE_loss(y_1: Float[Array, "... dim"], y_2: Float[Array, "... dim"], n: int) -> Real:
    """Compute the mean squared error between the average $n$ first spike times
    of two batches of spike trains.

    **Arguments**:

    - `y_1`: A batch of spike trains of shape `[..., dim]`.
    - `y_2`: Another batch of spike trains of shape `[..., dim]`.
    - `n`: The number of first spikes to consider.

    **Returns**:

    - A real number representing the mean squared error between the average $n$ first spike times.
    """
    first_spikes = get_n_first_spikes(y_1, n)
    avg_first_spikes_1 = jnp.mean(first_spikes, axis=0)
    first_spikes = get_n_first_spikes(y_2, n)
    avg_first_spikes_2 = jnp.mean(first_spikes, axis=0)
    return jnp.mean((avg_first_spikes_1 - avg_first_spikes_2) ** 2)
