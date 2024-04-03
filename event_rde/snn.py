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

from .paths import SpikeTrain
from .solution import Solution


class NetworkState(eqx.Module):
    ts: Real[Array, " times"]
    ys: Float[Array, "times neurons 3"]
    tevents: eqxi.MaybeBuffer[Real[Array, " spikes"]]
    yevents: eqxi.MaybeBuffer[Float[Array, "spikes neurons 3"]]
    t0: Real
    y0: Float[Array, "neurons 3"]
    num_spikes: Int
    event_mask: List[bool]
    event_types: eqxi.MaybeBuffer[Bool[Array, "spikes neurons"]]
    key: Any


def buffers(state: NetworkState):
    assert type(state) is NetworkState
    return state.tevents, state.yevents, state.event_types


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
    diffusion_vf: Optional[Callable[..., Float[ArrayLike, "neurons 3 2*neurons"]]]

    def __init__(
        self,
        num_neurons: Int,
        intensity_fn: Callable[..., Float],
        v_reset: Float = 1.0,
        alpha: Float = 1e-2,
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
                mu1, mu2 = self.mu  # type: ignore
                v, i, _ = y
                v_out = mu1 * (i + ic - v)
                i_out = -mu2 * i
                s_out = self.intensity_fn(v)
                return jnp.array([v_out, i_out, s_out])

            return _vf(y, ic)

        self.drift_vf = drift_vf

        if diffusion:
            if sigma is None:
                sigma = jr.normal(sigma_key, (2, 2))
                sigma = jnp.dot(sigma, sigma.T)
                self.sigma = sigma

            def diffusion_vf(t, y, args):
                sigma_zero = jnp.zeros((3, 2, num_neurons))

                @jax.vmap
                def _vf(k):
                    return sigma_zero.at[:2, :, k : (k + 1)].set(self.sigma)

                return _vf(jnp.arange(self.num_neurons))

            self.diffusion_vf = diffusion_vf
        else:
            self.sigma = None
            self.diffusion_vf = None

        def cond_fn(state, y, n, **kwargs):
            return y[n, 2]

        self.cond_fn = [ft.partial(cond_fn, n=n) for n in range(self.num_neurons)]

    @eqx.filter_jit
    def __call__(self, input_current, t0, t1, ts, v0, i0, max_spikes, *, key):
        t0, t1 = float(t0), float(t1)
        key, init_key, bm_key = jr.split(key, 3)
        s0 = jnp.log(jr.uniform(init_key, (self.num_neurons,))) - self.alpha
        y0 = jnp.vstack([v0, i0, s0]).T
        ys = jnp.full((ts.shape[0], self.num_neurons, 3), jnp.inf)
        # ys = ys.at[0, :, :].set(y0)
        tevents = jnp.full((max_spikes,), jnp.inf)
        yevents = jnp.full((max_spikes, self.num_neurons, 3), jnp.inf)
        event_mask = jtu.tree_map(lambda _y: False, self.cond_fn)
        event_types = jnp.full((max_spikes, self.num_neurons), False)
        init_state = NetworkState(ts, ys, tevents, yevents, t0, y0, 0, event_mask, event_types, key)

        dt0 = 0.01
        vf = diffrax.ODETerm(self.drift_vf)
        if self.diffusion_vf is not None:
            bm = diffrax.VirtualBrownianTree(
                t0, t1, tol=dt0 / 2, shape=(2, self.num_neurons), key=bm_key
            )
            cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
            terms = diffrax.MultiTerm(vf, cvf)
        else:
            terms = vf
        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()
        w_update = self.w.at[self.network].set(0.0)

        @jax.vmap
        def trans_fn(y, w, ev, key):
            v, i, s = y
            v_out = v - jnp.where(ev, self.v_reset, 0.0)
            i_out = i + w
            s_out = jnp.where(ev, jnp.log(jr.uniform(key)) - self.alpha, s)
            return jnp.array([v_out, i_out, s_out])

        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            trans_key = jr.split(trans_key, self.num_neurons)

            _t0 = state.t0
            _y0 = state.y0
            t_seq = jnp.where(state.ts > _t0, state.ts, _t0)
            saveat = diffrax.SaveAt(ts=t_seq)
            sol = diffrax.diffeqsolve(
                terms,
                solver,
                _t0,
                t1,
                dt0,
                _y0,
                input_current,
                saveat=saveat,
                event=event,
            )

            assert sol.event_mask is not None
            event_mask = sol.event_mask

            event_types = state.event_types
            event_types = event_types.at[state.num_spikes].set(jnp.array(event_mask))

            assert sol.ts is not None
            tevent = sol.ts[-1]
            tevents = state.tevents
            tevents = tevents.at[state.num_spikes].set(tevent)

            assert sol.ys is not None
            yevent = sol.ys[-1].reshape((self.num_neurons, 3))
            event_idx = jnp.argmax(jnp.array(event_mask))
            w_update_row = jax.lax.dynamic_slice(
                w_update, (event_idx, 0), (1, self.num_neurons)
            ).reshape((-1,))
            # jax.debug.print("{x}", x=w_update_row)
            event_array = jnp.array(event_mask)

            ytrans = trans_fn(yevent, w_update_row, event_array, trans_key)
            # jax.debug.print("{x}", x=ytrans - yevent)
            yevents = state.yevents
            yevents = yevents.at[state.num_spikes].set(yevent)

            num_spikes = state.num_spikes + 1

            update_mask = jnp.array((t_seq > _t0) & (t_seq <= tevent)).reshape((-1,))
            update_mask = jnp.tile(update_mask, (3, self.num_neurons, 1)).T
            ys = sol.ys
            ys = jnp.where(update_mask, ys, state.ys)
            ys = ys.at[jnp.sum(state.ts < _t0)].set(_y0)  # pyright: ignore

            new_state = NetworkState(
                ts=state.ts,
                ys=ys,
                tevents=tevents,
                yevents=yevents,
                t0=tevent,
                y0=ytrans,
                num_spikes=num_spikes,
                event_mask=event_mask,
                event_types=event_types,
                key=new_key,
            )

            return new_state

        def stop_fn(state: NetworkState) -> bool:
            return (state.num_spikes <= max_spikes) & (state.t0 < t1)

        final_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes,
            kind="checkpointed",
        )

        ys = final_state.ys
        ts = final_state.ts
        spike_times = final_state.tevents
        spike_marks = final_state.event_types
        spike_values = final_state.yevents
        num_spikes = final_state.num_spikes
        spike_train = SpikeTrain(t0, t1, spike_times, spike_marks)
        sol = Solution(
            ys=ys,
            ts=ts,
            spike_times=spike_times,
            spike_marks=spike_marks,
            spike_values=spike_values,
            spike_train=spike_train,
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
        key: Any,
        return_type: str = "spike_train",
    ):
        nn_minus_input = self.width_size * (self.depth - 1) + self.out_size
        nn_minus_output = nn_minus_input - self.out_size + self.in_size

        def _input_current(t: Float) -> Array:
            return jnp.hstack([input_current(t), jnp.zeros((self.num_neurons - self.in_size,))])

        out = super().__call__(_input_current, t0, t1, ts, v0, i0, max_spikes, key=key)
        if return_type == "spike_train":
            spike_trains = jnp.array(jax.vmap(out.spike_train.evaluate)(ts))
            spike_trains = spike_trains[:, nn_minus_output:]
            return spike_trains
        elif return_type == "solution":
            return out
