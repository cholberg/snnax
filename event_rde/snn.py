import functools as ft
from typing import Any, Callable, List, Optional

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
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


def _build_w(w, network, key):
    if w is not None:
        return w
    w_a = jr.uniform(key, network.shape, minval=0.5)
    return w_a.at[network].set(0.0)


def _inner_trans_fn(ev_outer, w, y, event_mask, key, v_reset):
    def _for_inner(_w, _y, ev_inner, ev_outer):
        v, i, s = _y
        s_key = jr.fold_in(key, v)
        v_out = jnp.where(ev_inner, v - v_reset, v)
        i_out = jnp.where(ev_inner, i, i + _w)
        s_out = jnp.where(ev_inner, jnp.log(jr.uniform(s_key)) - 1e-2, s)
        y_out = jnp.array([v_out, i_out, s_out])
        return jnp.where(ev_outer, y_out, jnp.full_like(y_out, 0.0))

    def _for_inner_fill(_y, ev_inner, ev_outer):
        v, i, s = _y
        s_key = jr.fold_in(key, v)
        v_out = jnp.where(ev_inner, v - v_reset, v)
        s_out = jnp.where(ev_inner, jnp.log(jr.uniform(s_key)) - 1e-2, s)
        y_out = jnp.array([v_out, i, s_out])
        return jnp.where(ev_outer, y_out, jnp.full_like(y_out, 0.0))

    out = jtu.tree_map(ft.partial(_for_inner, ev_outer=ev_outer), w, y, event_mask)
    out_fill = jtu.tree_map(ft.partial(_for_inner_fill, ev_outer=ev_outer), y, event_mask)
    out = eqx.combine(out, out_fill)
    return out


class SpikingNeuralNet(eqx.Module):
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
        mu: Optional[Float[ArrayLike, " 2"]] = None,
        diffusion: bool = False,
        sigma: Optional[Float[ArrayLike, "2 2"]] = None,
        key: Optional[Any] = None,
    ):
        self.num_neurons = num_neurons
        self.intensity_fn = intensity_fn
        self.v_reset = v_reset
        self.alpha = alpha

        if key is None:
            key = jax.random.PRNGKey(0)

        w_key, mu_key, sigma_key = jr.split(key, 3)

        if network is None:
            network = np.full((num_neurons, num_neurons), False)

        self.w = _build_w(w, network, w_key)
        self.network = network

        if mu is None:
            mu = jr.uniform(mu_key, (2,), minval=0.5)

        self.mu = mu

        def drift_vf(t, y, input_current):
            ic = input_current(t)

            @jax.vmap
            def _vf(y, ic):
                # y = eqx.error_if(y, jnp.any(jnp.isnan(y) | jnp.isinf(y)), f"{y}")
                mu1, mu2 = self.mu  # type: ignore
                v, i, _ = y
                v_out = mu1 * (i + ic - v)
                i_out = -mu2 * i
                s_out = self.intensity_fn(v)
                out = jnp.array([v_out, i_out, s_out])
                # out = eqx.error_if(out, jnp.any(jnp.isnan(out) | jnp.isinf(out)), "dxdt")
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
        t0, t1 = float(t0), float(t1)
        _t0 = jnp.full((num_samples,), t0)
        key, bm_key, init_key = jr.split(key, 3)
        s0_key, i0_key, v0_key = jr.split(init_key, 3)
        s0 = jnp.log(jr.uniform(s0_key, (num_samples, self.num_neurons))) - self.alpha
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
        solver = diffrax.Heun()
        w_update = self.w.at[self.network].set(0.0)
        # bm_key is not updated in body_fun since we want to make sure that the same Brownian path
        # is used for before and after each spike.
        bm_key = jr.split(bm_key, num_samples)

        @jax.vmap
        def trans_fn(y, w, ev, key):
            v, i, s = y
            v_out = v - jnp.where(ev, self.v_reset, 0.0)
            i_out = i + w
            s_out = jnp.where(ev, jnp.log(jr.uniform(key)) - self.alpha, s)
            # ensures that s_out does not exceed 0 in cases where two events are triggered
            s_out = jnp.minimum(s_out, -1e-3)
            return jnp.array([v_out, i_out, s_out])

        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            trans_key = jr.split(trans_key, num_samples)

            @jax.vmap
            def update(_t0, y0, trans_key, bm_key):
                ts = jnp.where(
                    _t0 < t1 - 1 / num_save,
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
                tevent = jnp.abs(sol.ts[-1])  # if t0 == t1 this would otherwise be negative

                assert sol.ys is not None
                ys = jnp.where(_t0 < t1, sol.ys, jnp.tile(y0[None, :], (num_save, 1, 1)))
                yevent = ys[-1].reshape((self.num_neurons, 3))
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan")
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf")
                event_idx = jnp.argmax(jnp.array(event_mask))
                w_update_row = jax.lax.dynamic_slice(
                    w_update, (event_idx, 0), (1, self.num_neurons)
                ).reshape((-1,))
                w_update_row = jnp.where(event_happened, w_update_row, 0.0)
                ytrans = trans_fn(yevent, w_update_row, event_array, trans_key)
                ys = ys.reshape((self.num_neurons, num_save, 3))  # pyright: ignore

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

            """update_mask = jnp.array((t_seq > _t0) & (t_seq <= tevent)).reshape((-1,))
            update_mask = jnp.tile(update_mask, (3, self.num_neurons, 1)).T
            ys = sol.ys
            ys = jnp.where(update_mask, ys, state.ys)
            ys = ys.at[jnp.sum(state.ts < _t0)].set(_y0)  # pyright: ignore"""

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
    layer_idx = np.cumsum(jnp.array(layer_idx))
    for i in range(depth):
        lrows = layer_idx[i]
        urows = layer_idx[i + 1]
        lcols = layer_idx[i + 1]
        ucols = layer_idx[i + 2]
        network_fill = np.full((urows - lrows, ucols - lcols), False)
        network_out[lrows:urows, lcols:ucols] = network_fill
    return network_out


class FeedForwardSNN(SpikingNeuralNet):
    in_size: Int
    out_size: Int
    width_size: Int
    depth: Int

    def __init__(self, in_size, out_size, width_size, depth, intensity_fn, key, diffusion=False):
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        num_neurons = self.in_size + self.width_size * (self.depth - 1) + self.out_size
        network = _build_forward_network(self.in_size, self.out_size, self.width_size, self.depth)
        super().__init__(num_neurons, intensity_fn, network=network, key=key, diffusion=diffusion)

    def __call__(
        self,
        input_current: Callable[..., Float[Array, " input_size"]],
        t0: Real,
        t1: Real,
        ts: Float[ArrayLike, ""],
        v0: Float[ArrayLike, " neurons"],
        i0: Float[ArrayLike, " neurons"],
        max_spikes: int,
        num_samples: int,
        key: Any,
        num_save: int = 2,
    ):
        def _input_current(t: Float) -> Array:
            return jnp.hstack([input_current(t), jnp.zeros((self.num_neurons - self.in_size,))])

        return super().__call__(
            _input_current, t0, t1, max_spikes, num_samples, key=key, num_save=num_save
        )
