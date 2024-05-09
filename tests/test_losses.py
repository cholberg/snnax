import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from event_rde import (
    SpikingNeuralNet,
    marcus_lift,
    signature_mmd,
)

SEED = 1234
key = jr.PRNGKey(SEED)
key, init_key, dat_key, scales_key = jr.split(key, 4)
snn_init_params = {
    "num_neurons": 1,
    "v_reset": 1.4,
    "alpha": 3e-2,
    "w": jnp.array([[0.0]]),
    "network": jnp.array([[True]]),
    "mu": np.array([15, 0.0]),
    "sigma": jnp.array([[0.25, 0.0], [0.0, 0.0]]),
    "key": init_key,
    "diffusion": False,
}
snn_call_params = {
    "t0": 0,
    "t1": 2,
    "max_spikes": 3,
    "num_samples": 1,
    "v0": jnp.full((1, 1), 0),
    "i0": jnp.full((1, 1), 0),
    "key": dat_key,
    "dt0": 1e-1,
}
extra_params = {"c": 1.5, "tau_s": 1, "beta": 5, "v_th": 1}


@jax.vmap
def get_marcus_lifts(spike_times, spike_marks):
    return marcus_lift(snn_call_params["t1"], snn_call_params["t1"], spike_times, spike_marks[:, :])


def generate_data(c):
    def intensity_fn(v: Float) -> Float:
        return jnp.exp(extra_params["beta"] * (v - extra_params["v_th"])) / extra_params["tau_s"]

    def input_current(t: Float) -> Array:
        return jnp.array([c])

    snn_true = SpikingNeuralNet(
        **snn_init_params,
        intensity_fn=intensity_fn,
    )
    sol = snn_true(input_current, **snn_call_params)
    spikes = get_marcus_lifts(sol.spike_times, sol.spike_marks)
    return spikes


def test_sig_mmd():
    spike_true = generate_data(extra_params["c"])
    scales = jax.random.exponential(scales_key, shape=(1,))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss(c):
        spike_pred = generate_data(c)
        return signature_mmd(spike_true, spike_pred, scales=scales, refinement_factor=1)

    c = jnp.array(1.0)
    val, grad = loss(c)
    assert not (jnp.isnan(val) | jnp.isinf(val))
    assert not (jnp.isnan(grad) | jnp.isinf(grad))
