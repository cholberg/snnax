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
    sig_1 = expected_signature(y_1, depth)
    sig_2 = expected_signature(y_2, depth)
    return jnp.mean((sig_1 - sig_2) ** 2)
