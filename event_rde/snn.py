import functools as ft
from typing import Any, Callable, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int, PyTree, Real


def _is_none(x):
    return x is None


class NetworkState(eqx.Module):
    ts: Array
    ys: PyTree[Array]
    tevents: Array
    yevents: PyTree[Array]
    t0: Real
    y0: PyTree[ArrayLike]
    num_spikes: Int
    event_mask: PyTree[bool]
    event_types: PyTree[Array]
    key: Any


class SpikingNeuralNet(eqx.Module):
    neurons: PyTree[Int, " neurons"]
    num_neurons: Int
    w: Float[Array, "neurons neurons"]
    v_reset: Float
    alpha: Float
    mu: Float[Array, " dim2"]
    drift_vf: Callable[..., PyTree[Float[Array, " dim3"], " neurons"]]
    cond_fn: PyTree[Callable[..., Float], " neurons"]
    intensity_fn: Callable[..., Float]
    sigma: Optional[Float[Array, "dim2 dim2"]] = None
    diffusion_vf: Optional[Callable[..., PyTree[Float[Array, "dim3 dim2"], " neurons"]]] = None

    def __init__(
        self,
        w,
        neurons,
        v_reset,
        alpha,
        mu,
        intensity_fn,
        sigma=None,
    ):
        self.neurons = neurons
        self.w = w
        self.num_neurons = self.w.shape[0]
        self.v_reset = v_reset
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.intensity_fn = intensity_fn

        def drift_vf(t, y, input_current):
            def _vf(y, input_current):
                mu1, mu2 = self.mu
                v, i, s = y
                v_out = mu1 * (i - v)
                i_out = -mu2 * i
                if input is not None:
                    v_out = v_out + mu1 * input_current(t)
                s_out = self.intensity_fn(v)
                return jnp.array([v_out, i_out, s_out])

            return jtu.tree_map(_vf, y, input_current)

        self.drift_vf = drift_vf

        def cond_fn(state, y, path, **kwargs):
            for yp, yl in jtu.tree_leaves_with_path(y):
                if yp == path:
                    return yl[2]

        self.cond_fn = jtu.tree_map_with_path(lambda path, _: ft.partial(cond_fn, path=path), self.neurons)

        if self.sigma is not None:

            def diffusion_vf(t, y, args):
                return jtu.tree_map(lambda x: jnp.vstack([self.sigma, jnp.zeros((2,))]), y)  # pyright: ignore

            self.diffusion_vf = diffusion_vf

    def __call__(self, input_current, ts, v0, i0, max_spikes, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        key, init_key, bm_key = jr.split(key, 3)
        s0 = jtu.tree_map(
            lambda n: jnp.log(jr.uniform(jr.fold_in(init_key, n))) - self.alpha,
            self.neurons,
        )
        y0 = jtu.tree_map(lambda v, i, s: jnp.array([v, i, s]), v0, i0, s0)
        ys = jtu.tree_map(lambda _y: jnp.tile(jnp.full_like(_y, jnp.inf), (ts.shape[0], 1)), y0)
        ys = jtu.tree_map(lambda _ys, _y: _ys.at[0].set(_y), ys, y0)
        tevents = jnp.full((max_spikes,), jnp.inf)
        yevents = jtu.tree_map(lambda _y: jnp.tile(jnp.full_like(_y, jnp.inf), (max_spikes, 1)), y0)
        event_mask = jtu.tree_map(lambda _y: False, self.cond_fn)
        event_types = jtu.tree_map(lambda _: jnp.full((max_spikes,), False), self.neurons)
        init_state = NetworkState(ts, ys, tevents, yevents, t0, y0, 0, event_mask, event_types, key)

        dt0 = 0.01
        vf = diffrax.ODETerm(self.drift_vf)
        if self.diffusion_vf is not None:

            def shape_map(_y):
                return jtu.tree_map(lambda x: self.sigma[0, :], _y)  # pyright: ignore

            bm_shape = jax.eval_shape(shape_map, y0)
            bm = diffrax.VirtualBrownianTree(t0, t1, tol=dt0 / 2, shape=bm_shape, key=key)
            cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
            terms = diffrax.MultiTerm(vf, cvf)
        else:
            terms = vf

        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()

        def trans_fn(w, y, ev, key):
            key = jr.fold_in(key, w)
            v, i, s = y
            v_out = jnp.where(ev, v - self.v_reset, v)
            i_out = jnp.where(ev, i, i + w)
            s_out = jnp.where(ev, jnp.log(jr.uniform(key)) - self.alpha, s)
            y_out = jnp.array([v_out, i_out, s_out])
            return y_out

        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            new_state = eqx.tree_at(lambda s: s.key, state, new_key)

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

            event_mask = sol.event_mask
            event_types = state.event_types
            event_types = jtu.tree_map(lambda et, em: et.at[state.num_spikes].set(em), event_types, event_mask)
            new_state = eqx.tree_at(lambda s: s.event_mask, new_state, event_mask)
            new_state = eqx.tree_at(lambda s: s.event_types, new_state, event_types)

            assert sol.ts is not None
            tevent = sol.ts[-1]
            tevents = state.tevents
            tevents = tevents.at[state.num_spikes].set(tevent)
            new_state = eqx.tree_at(lambda s: s.t0, new_state, tevent)
            new_state = eqx.tree_at(lambda s: s.tevents, new_state, tevents)

            yevent = jtu.tree_map(lambda _y: _y[-1], sol.ys)
            event_idx = jnp.argmax(jnp.array(jtu.tree_leaves(event_mask)))
            w_update = jax.lax.dynamic_slice(self.w, (event_idx, 0), (1, self.num_neurons)).reshape((-1,))
            w_update = jtu.build_tree(jtu.tree_structure(self.neurons), w_update)
            ytrans = jtu.tree_map(ft.partial(trans_fn, key=trans_key), w_update, yevent, event_mask)
            yevents = state.yevents
            yevents = jtu.tree_map(lambda y, _y: y.at[state.num_spikes].set(_y), yevents, yevent)
            new_state = eqx.tree_at(lambda s: s.y0, new_state, ytrans)
            new_state = eqx.tree_at(lambda s: s.yevents, new_state, yevents)

            num_spikes = state.num_spikes + 1
            new_state = eqx.tree_at(lambda s: s.num_spikes, new_state, num_spikes)

            update_mask = jnp.array((t_seq > _t0) & (t_seq <= tevent)).reshape((-1,))
            update_mask = jnp.tile(update_mask, (3, 1)).T
            ys = jtu.tree_map(lambda _ys, __ys: jnp.where(update_mask, __ys, _ys), state.ys, sol.ys)
            new_state = eqx.tree_at(lambda s: s.ys, new_state, ys)

            return new_state

        def stop_fn(state: NetworkState) -> bool:
            return (state.num_spikes <= max_spikes) & (state.t0 < t1)

        final_state = jax.lax.while_loop(stop_fn, body_fun, init_state)

        return final_state
