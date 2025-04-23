import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from snnax import (
    SpikingNeuralNet,
)

key = jr.PRNGKey(12345)

key, init_key, dat_key, scales_key = jr.split(key, 4)
snn_init_params = {
    "num_neurons": 1,
    "v_reset": 1.4,  # Amount to reset voltage to after a spike
    "alpha": 3e-2,  # Variable controlling refractory period
    "w": jnp.array([[0.0]]),  # Weight matrix (trivial in the case of a single neuron)
    "network": np.array([[True]]),
    "mu": np.array([10, 0.0]),  # Variable in LIF drift term
    "sigma": jnp.array([[0.5, 0.0], [0.0, 0.0]]),  # Diffusion matrix
    "key": init_key,
    "diffusion": False,  # Whether to include diffusion in the LIF SDE
}
snn_call_params = {
    "t0": 0,
    "t1": 2,
    "max_spikes": 10,  # Maximum number of spikes allowed
    "num_samples": 10,  # Number of samples to generate
    "num_save": 1000,  # Number of time points to save per spike
    "key": dat_key,
    "dt0": 1e-2,
}
extra_params = {
    "c": 1.2,  # Input current
    "tau_s": 1,  # Variable in intensity function
    "beta": 20,  # Varibale in intensity function
    "v_th": 1,  # Spike treshold
}


# Intensity function for spike triggering
def intensity_fn(v: Float) -> Float:
    return jnp.exp(extra_params["beta"] * (v - extra_params["v_th"])) / extra_params["tau_s"]


def test_snn():
    snn = SpikingNeuralNet(
        **snn_init_params,
        intensity_fn=intensity_fn,
    )

    def input_current(t: Float) -> Array:
        return jnp.array([extra_params["c"]])

    snn_call_params["v0"] = np.full(
        (snn_call_params["num_samples"], snn_init_params["num_neurons"]), 0
    )
    snn_call_params["i0"] = np.full(
        (snn_call_params["num_samples"], snn_init_params["num_neurons"]), 0
    )
    sol = snn(input_current, **snn_call_params)

    ys = sol.ys
    ts = sol.ts
    st = sol.spike_times
    sm = sol.spike_marks[:, :, 0]

    assert ys.shape == (10, 10, 1, 1000, 3)
    assert st.shape == (10, 10)
    ys = ys[:, :, 0]

    # Check correct initialization
    assert jnp.all(ys[:, 0, 0, :2] == 0)
    assert jnp.all((ys[:, :, 0, 2] <= 0) | jnp.isinf(ys[:, :, 0, 2]))

    # Check correct termination
    spike_idx = jnp.sum(sol.ts < jnp.inf, axis=-1) - 1
    i = jnp.arange(10)[:, None]
    j = jnp.arange(10)[None, :]
    spike_ys = ys[i, j, spike_idx, :]
    spike_ts = ts[i, j, spike_idx]

    assert jnp.nanmax(jnp.abs(spike_ts - sol.spike_times)) < 1e-2
    assert jnp.all(spike_ys[:, :, 2][sm] > -1e-1)


def test_ssnn():
    # Add diffusion term
    _snn_init_params = snn_init_params.copy()
    _snn_init_params["diffusion"] = True
    snn = SpikingNeuralNet(
        **_snn_init_params,
        intensity_fn=intensity_fn,
    )

    def input_current(t: Float) -> Array:
        return jnp.array([extra_params["c"]])

    snn_call_params["v0"] = np.full(
        (snn_call_params["num_samples"], snn_init_params["num_neurons"]), 0
    )
    snn_call_params["i0"] = np.full(
        (snn_call_params["num_samples"], snn_init_params["num_neurons"]), 0
    )
    sol = snn(input_current, **snn_call_params)

    ys = sol.ys
    ts = sol.ts
    st = sol.spike_times
    sm = sol.spike_marks[:, :, 0]

    assert ys.shape == (10, 10, 1, 1000, 3)
    assert st.shape == (10, 10)
    ys = ys[:, :, 0]

    # Check correct initialization
    assert jnp.all(ys[:, 0, 0, :2] == 0)
    assert jnp.all((ys[:, :, 0, 2] <= 0) | jnp.isinf(ys[:, :, 0, 2]))

    # Check correct termination
    spike_idx = jnp.sum(sol.ts < jnp.inf, axis=-1) - 1
    i = jnp.arange(10)[:, None]
    j = jnp.arange(10)[None, :]
    spike_ys = ys[i, j, spike_idx, :]
    spike_ts = ts[i, j, spike_idx]

    assert jnp.nanmax(jnp.abs(spike_ts - sol.spike_times)) < 1e-2
    assert jnp.all(spike_ys[:, :, 2][sm] > -1)
