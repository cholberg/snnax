import functools as ft
from typing import Any, Callable, List, Optional

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, Real

from .paths import BrownianPath
from .solution import Solution


class NetworkState(eqx.Module):
    ts: Real[Array, "samples spikes times"]
    ys: Float[Array, "samples spikes neurons times 3"]
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 3"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]
    key: Any


def buffers(state: NetworkState):
    assert type(state) is NetworkState
    return state.tevents, state.ts, state.ys, state.event_types


def _build_w(w, network, key, minval, maxval):
    if w is not None:
        return w
    w_a = jr.uniform(key, network.shape, minval=minval, maxval=maxval)
    return w_a.at[network].set(0.0)


class SpikingNeuralNet(eqx.Module):
    """A class representing a generic stochastic spiking neural network."""

    num_neurons: Int
    w: Float[Array, "neurons neurons"]
    network: Bool[ArrayLike, "neurons neurons"] = eqx.field(static=True)
    v_reset: Float
    alpha: Float
    mu: Float[ArrayLike, " 2"]
    drift_vf: Callable[..., Float[ArrayLike, "neurons 3"]]
    cond_fn: List[Callable[..., Float]]
    intensity_fn: Callable[..., Float]
    sigma: Optional[Float[ArrayLike, "2 2"]]
    diffusion_vf: Optional[Callable[..., Float[ArrayLike, "neurons 3 2 neurons"]]]

    def __init__(
        self,
        num_neurons: Int,
        intensity_fn: Callable[..., Float],
        v_reset: Float = 1.0,
        alpha: Float = 3e-2,
        w: Optional[Float[Array, "neurons neurons"]] = None,
        network: Optional[Bool[ArrayLike, "neurons neurons"]] = None,
        wmin: Float = 0.5,
        wmax: Float = 1.0,
        mu: Optional[Float[ArrayLike, " 2"]] = None,
        diffusion: bool = False,
        sigma: Optional[Float[ArrayLike, "2 2"]] = None,
        key: Optional[Any] = None,
    ):
        """**Arguments**:

        - `num_neurons`: The number of neurons in the network.
        - `intensity_fn`: The intensity function for spike generation.
            Should take as input a scalar (voltage) and return a scalar (intensity).
        - `v_reset`: The reset voltage value for neurons. Defaults to 1.0.
        - `alpha`: Constant controlling the refractory period. Defaults to 3e-2.
        - `w`: The initial weight matrix. Should be a square matrix of size `num_neurons`.
            If none, the weights are randomly intialized for all entries in `self.network`
            that are False.
        - `network`: The connectivity matrix of the network. Shuold be a square matrix of size
            `num_neurons` with the $ij$'th element being `False` if there is no connection from
            neuron $i$ to neuron $j$. If none is provided, the network is fully connected.
        - `wmin`: The minimum weight value for random initialization. Defaults to 0.5.
        - `wmax`: The maximum weight value for random initialization. Defaults to 1.0.
        - `mu`: A 2-dimensional vector describing the drift term of each neuron.
            If none is provided, the values are randomly initialized.
        - `diffusion`: Whether to include diffusion term in the SDE. Defaults to False.
        - `sigma`: A 2 by 2 diffusion matrix. If none is provided, the values are randomly
            initialized.
        - `key`: The random key for initialization. If None,
            the key is set to `jax.random.PRNGKey(0)`.
        """

        self.num_neurons = num_neurons
        self.intensity_fn = intensity_fn
        self.v_reset = v_reset
        self.alpha = alpha

        if key is None:
            key = jax.random.PRNGKey(0)

        w_key, mu_key, sigma_key = jr.split(key, 3)

        if network is None:
            network = np.full((num_neurons, num_neurons), False)

        self.w = _build_w(w, network, w_key, wmin, wmax)
        self.network = network

        if mu is None:
            mu = jr.uniform(mu_key, (2,), minval=0.5)

        self.mu = mu

        def drift_vf(t, y, input_current):
            ic = input_current(t)

            @jax.vmap
            def _vf(y, ic):
                mu1, mu2 = self.mu  # type: ignore
                v, i, _ = y
                v_out = mu1 * (i + ic - v)
                i_out = -mu2 * i
                s_out = self.intensity_fn(v)
                out = jnp.array([v_out, i_out, s_out])
                return out

            return _vf(y, ic)

        self.drift_vf = drift_vf

        if diffusion:
            if sigma is None:
                sigma = jr.normal(sigma_key, (2, 2))
                sigma = jnp.dot(sigma, sigma.T)
                self.sigma = sigma

            sigma_large = jnp.zeros((num_neurons, 3, 2, num_neurons))
            for k in range(num_neurons):
                sigma_large = sigma_large.at[k, :2, :, k].set(sigma)

            def diffusion_vf(t, y, args):
                return sigma_large

            self.diffusion_vf = diffusion_vf
            self.sigma = sigma
        else:
            self.sigma = None
            self.diffusion_vf = None

        def cond_fn(state, y, n, **kwargs):
            return y[n, 2]

        self.cond_fn = [ft.partial(cond_fn, n=n) for n in range(self.num_neurons)]

    @eqx.filter_jit
    def __call__(
        self,
        input_current,
        t0,
        t1,
        max_spikes,
        num_samples,
        *,
        key,
        v0=None,
        i0=None,
        num_save=2,
        dt0=0.01,
        max_steps=1000,
    ):
        """**Arguments:**

            `input_current`: The input current to the SNN model. Should be a function
                taking as input a scalar time value and returning a vector of size
                `self.num_neurons`.
            `t0`: The starting time of the simulation.
            `t1`: The ending time of the simulation.
            `max_spikes`: The maximum number of spikes allowed in the simulation.
            `num_samples`: The number of samples to simulate.
            `key`: The random key used for generating random numbers.
            `v0`: The initial membrane potential of the neurons. If None,
                it will be randomly generated. Otherwise, it should be a vector of shape
                `(num_samples, self.num_neurons)`.
            `i0`: The initial membrane potential of the neurons. If None,
                it will be randomly generated. Otherwise, it should be a vector of shape
                `(num_samples, self.num_neurons)`.
            `num_save`: The number of time points to save per spike during the simulation.
            `dt0`: The time step size used in the differential equation solve.
            `max_steps`: The maximum number of steps allowed in the differential equation solve.

        **Returns:**

            `Solution`: An object containing the simulation results,
                including the time points, membrane potentials,
                 spike times, spike marks, and the number of spikes.
        """

        t0, t1 = float(t0), float(t1)
        _t0 = jnp.full((num_samples,), t0)
        key, bm_key, init_key = jr.split(key, 3)
        s0_key, i0_key, v0_key = jr.split(init_key, 3)
        # to ensure that s0 != -inf, we set minval=1e-10
        s0 = jnp.log(jr.uniform(s0_key, (num_samples, self.num_neurons), minval=1e-10)) - self.alpha
        s0 = eqx.error_if(s0, jnp.any(jnp.isinf(s0)), "s0 is inf")
        if v0 is None:
            v0 = jr.uniform(
                v0_key, (num_samples, self.num_neurons), minval=0.0, maxval=self.v_reset
            )
        if i0 is None:
            i0 = jr.uniform(
                i0_key, (num_samples, self.num_neurons), minval=0.0, maxval=self.v_reset
            )
        y0 = jnp.dstack([v0, i0, s0])
        ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 3), jnp.inf)
        ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = 0
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)
        init_state = NetworkState(
            ts, ys, tevents, _t0, y0, num_spikes, event_mask, event_types, key
        )

        # stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        vf = diffrax.ODETerm(self.drift_vf)
        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()
        w_update = self.w.at[self.network].set(0.0)
        # bm_key is not updated in body_fun since we want to make sure that the same Brownian path
        # is used for before and after each spike.
        bm_key = jr.split(bm_key, num_samples)

        @jax.vmap
        def trans_fn(y, w, ev, key):
            v, i, s = y
            v_out = v - jnp.where(ev, self.v_reset, 0.0)
            i_out = i + w
            s_out = jnp.where(ev, jnp.log(jr.uniform(key, minval=1e-10)) - self.alpha, s)
            # ensures that s_out does not exceed 0 in cases where two events are triggered
            s_out = jnp.minimum(s_out, -1e-3)
            return jnp.array([v_out, i_out, s_out])

        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            trans_key = jr.split(trans_key, num_samples)

            @jax.vmap
            def update(_t0, y0, trans_key, bm_key):
                ts = jnp.where(
                    _t0 < t1 - (t1 - t0) / (10 * num_save),
                    jnp.linspace(_t0, t1, num_save),
                    jnp.full((num_save,), _t0),
                )
                ts = eqxi.error_if(ts, ts[1:] < ts[:-1], "ts must be increasing")
                trans_key = jr.split(trans_key, self.num_neurons)
                saveat = diffrax.SaveAt(ts=ts)
                terms = vf
                if self.diffusion_vf is not None:
                    bm = BrownianPath(
                        t0 - 1, t1 + 1, tol=dt0 / 2, shape=(2, self.num_neurons), key=bm_key
                    )
                    cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
                    terms = diffrax.MultiTerm(terms, cvf)
                sol = diffrax.diffeqsolve(
                    terms,
                    solver,
                    _t0,
                    t1,
                    dt0,
                    y0,
                    input_current,
                    throw=True,
                    # stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                )

                assert sol.event_mask is not None
                event_mask = jnp.array(sol.event_mask)
                event_happened = jnp.any(event_mask)
                event_array = jnp.array(event_mask)

                assert sol.ts is not None
                ts = sol.ts
                # Diffrax flips the sign of ts when t0 >= t1
                ts = jnp.where(t1 <= _t0, -ts, ts)
                tevent = ts[-1]
                tevent = eqxi.error_if(tevent, jnp.isnan(tevent), "tevent is nan")

                assert sol.ys is not None
                ys = sol.ys
                yevent = ys[-1].reshape((self.num_neurons, 3))
                yevent = jnp.where(_t0 < t1, yevent, y0)
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan")
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf")
                event_idx = jnp.argmax(jnp.array(event_mask))
                w_update_row = jax.lax.dynamic_slice(
                    w_update, (event_idx, 0), (1, self.num_neurons)
                ).reshape((-1,))
                w_update_row = jnp.where(event_happened, w_update_row, 0.0)
                ytrans = trans_fn(yevent, w_update_row, event_array, trans_key)
                ys = jnp.transpose(ys, (1, 0, 2))

                return ts, ys, tevent, ytrans, event_array

            _ts, _ys, tevent, _ytrans, event_mask = update(state.t0, state.y0, trans_key, bm_key)
            num_spikes = state.num_spikes + 1

            ts = state.ts
            ts = ts.at[:, state.num_spikes].set(_ts)

            ys = state.ys
            ys = ys.at[:, state.num_spikes].set(_ys)

            event_types = state.event_types
            event_types = event_types.at[:, state.num_spikes].set(event_mask)

            tevents = state.tevents
            tevents = tevents.at[:, state.num_spikes].set(tevent)

            new_state = NetworkState(
                ts=ts,
                ys=ys,
                tevents=tevents,
                t0=tevent,
                y0=_ytrans,
                num_spikes=num_spikes,
                event_mask=event_mask,
                event_types=event_types,
                key=new_key,
            )

            return new_state

        def stop_fn(state: NetworkState) -> bool:
            return (jnp.max(state.num_spikes) <= max_spikes) & (jnp.min(state.t0) < t1)

        final_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes,
            kind="checkpointed",
        )

        # ys = final_state.ys
        ys = final_state.ys
        ts = final_state.ts
        spike_times = final_state.tevents
        spike_marks = final_state.event_types
        num_spikes = final_state.num_spikes
        sol = Solution(
            t1=t1,
            ys=ys,
            ts=ts,
            spike_times=spike_times,
            spike_marks=spike_marks,
            num_spikes=num_spikes,
            max_spikes=max_spikes,
        )
        return sol


def _build_forward_network(in_size, out_size, width_size, depth):
    if depth <= 1:
        width_size = out_size
    num_neurons = in_size + width_size * (depth - 1) + out_size
    network_out = np.full((num_neurons, num_neurons), True)
    layer_idx = [0] + [in_size] + [width_size] * (depth - 1) + [out_size]
    layer_idx = np.cumsum(np.array(layer_idx))
    for i in range(depth):
        lrows = layer_idx[i]
        urows = layer_idx[i + 1]
        lcols = layer_idx[i + 1]
        ucols = layer_idx[i + 2]
        network_fill = np.full((urows - lrows, ucols - lcols), False)
        network_out[lrows:urows, lcols:ucols] = network_fill
    return network_out


class FeedForwardSNN(SpikingNeuralNet):
    """A convenience wrapper around `SpikingNeuralNet` for a feedforward network."""

    in_size: Int
    out_size: Int
    width_size: Int
    depth: Int

    def __init__(self, in_size, out_size, width_size, depth, intensity_fn, key, **kwargs):
        """**Arguments**:

        - `in_size`: The number of input neurons.
        - `out_size`: The number of output neurons.
        - `width_size`: The number of neurons in each hidden layer.
        - `depth`: The number of hidden layers.
        - `intensity_fn`: The intensity function for spike generation.
            Should take as input a scalar (voltage) and return a scalar (intensity).
        - `key`: The random key for initialization.
        - `**kwargs`: Additional keyword arguments passed to `SpikingNeuralNet`.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        num_neurons = self.in_size + self.width_size * (self.depth - 1) + self.out_size
        network = _build_forward_network(self.in_size, self.out_size, self.width_size, self.depth)
        super().__init__(
            num_neurons=num_neurons, intensity_fn=intensity_fn, network=network, key=key, **kwargs
        )

    def __call__(
        self,
        input_current: Callable[..., Float[Array, " input_size"]],
        t0: Real,
        t1: Real,
        max_spikes: int,
        num_samples: int,
        *,
        v0: Real[ArrayLike, " neurons"],
        i0: Real[ArrayLike, " neurons"],
        key: Any,
        num_save: int = 2,
        dt0: Real = 0.01,
    ):
        """**Arguments**:

        - `input_current`: The input current to the SNN model. Should be a function
            taking as input a scalar time value and returning a vector of size
            `self.in_size`.
        - `t0`: The starting time of the simulation.
        - `t1`: The ending time of the simulation.
        - `max_spikes`: The maximum number of spikes allowed in the simulation.
        - `num_samples`: The number of samples to simulate.
        - `key`: The random key used for generating random numbers.
        - `v0`: The initial membrane potential of the neurons. Should be a vector of shape
            `(num_samples, self.num_neurons)`.
        - `i0`: The initial membrane potential of the neurons. Should be a vector of shape
            `(num_samples, self.num_neurons)`.
        - `num_save`: The number of time points to save per spike during the simulation.
        - `dt0`: The time step size used in the differential equation solve.

        **Returns:**

            `Solution`: An object containing the simulation results,
                including the time points, membrane potentials,
                 spike times, spike marks, and the number of spikes.
        """

        def _input_current(t: Float) -> Array:
            return jnp.hstack([input_current(t), jnp.zeros((self.num_neurons - self.in_size,))])

        return super().__call__(
            _input_current,
            t0,
            t1,
            max_spikes,
            num_samples,
            key=key,
            v0=v0,
            i0=i0,
            num_save=num_save,
            dt0=dt0,
        )
