import functools as ft

import jax
import jax.numpy as jnp
import signax
from jaxtyping import Array, Float, Real


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
