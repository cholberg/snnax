import functools as ft
from typing import Any, Callable, Optional, Tuple, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int, PyTree, Real

from .paths import SpikeTrain
from .solution import Solution


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


def _is_none(x):
    return x is None


def get_switch(collection, idx, flatten_one=False):
    if flatten_one:
        collection, _ = eqx.tree_flatten_one_level(collection)
    funcs = [lambda i=i: collection[i] for i in range(len(collection))]
    return jax.lax.switch(idx, funcs)


def _build_w(w, neurons, key):
    num_neurons = len(neurons)
    w_a = jr.uniform(key, (num_neurons, num_neurons), minval=0.5)
    if w is None:
        w_d = tuple(tuple(w_a[i, j] for j in neurons) for i in neurons)
    elif isinstance(jtu.tree_leaves(w)[0], bool):
        w_d = tuple(tuple(w_a[i, j] if w[i][j] else None for j in neurons) for i in neurons)
    else:
        w_d = w
    return w_d


def _build_initial(x, neurons, key):
    if x is None:
        out = jtu.tree_map(lambda n: jr.uniform(jr.fold_in(key, n), maxval=0.1), neurons)
    else:
        out = jtu.tree_map(
            lambda n, _x: jr.uniform(jr.fold_in(key, n), maxval=0.1) if _x is None else _x,
            neurons,
            x,
        )
    return out


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
    neurons: Tuple[Int, ...]
    num_neurons: Int
    w: Tuple[Tuple[Optional[Float], ...], ...]
    v_reset: Float
    alpha: Float
    v0: Tuple[Float, ...]
    i0: Tuple[Float, ...]
    mu: Float[Array, " 2"]
    drift_vf: Callable[..., Tuple[Float[Array, " 3"], ...]]
    cond_fn: Tuple[Callable[..., Float], ...]
    intensity_fn: Callable[..., Float]
    sigma: Optional[Float[Array, "2 2"]]
    diffusion_vf: Optional[Callable[..., Tuple[Float[Array, "3 2"], ...]]]

    def __init__(
        self,
        num_neurons: Int,
        intensity_fn: Callable[..., Float],
        v_reset: Float = 1.0,
        alpha: Float = 1e-2,
        v0: Optional[Tuple[Optional[Float], ...]] = None,
        i0: Optional[Tuple[Optional[Float], ...]] = None,
        w: Optional[
            Union[Tuple[Tuple[Optional[Float], ...], ...], Tuple[Tuple[bool, ...], ...]]
        ] = None,
        mu: Optional[Float[Array, " 2"]] = None,
        diffusion: bool = False,
        sigma: Optional[Float[Array, "2 2"]] = None,
        key: Optional[Any] = None,
    ):
        self.num_neurons = num_neurons
        self.neurons = tuple(range(self.num_neurons))
        self.intensity_fn = intensity_fn
        self.v_reset = v_reset
        self.alpha = alpha

        if key is None:
            key = jax.random.PRNGKey(0)

        v0_key, i0_key, w_key, mu_key, sigma_key = jr.split(key, 5)
        self.v0 = _build_initial(v0, self.neurons, v0_key)
        self.i0 = _build_initial(i0, self.neurons, i0_key)
        self.w = _build_w(w, self.neurons, w_key)

        if mu is None:
            mu = jr.uniform(mu_key, (2,), minval=0.5)

        self.mu = mu

        if diffusion:
            if sigma is None:
                sigma = jr.normal(sigma_key, (2, 2))
                sigma = jnp.dot(sigma, sigma.T)

            def diffusion_vf(t, y, args):
                return jtu.tree_map(lambda x: jnp.vstack([sigma, jnp.zeros((2,))]), y)

            self.diffusion_vf = diffusion_vf
        else:
            sigma = None
            self.diffusion_vf = None

        self.sigma = sigma

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

        def cond_fn(state, y, n, **kwargs):
            return y[n][2]

        self.cond_fn = jtu.tree_map(lambda n: ft.partial(cond_fn, n=n), self.neurons)

    def __call__(self, input_current, ts, max_spikes, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        key, init_key, bm_key = jr.split(key, 3)
        s0 = jtu.tree_map(
            lambda n: jnp.log(jr.uniform(jr.fold_in(init_key, n))) - self.alpha, self.neurons
        )
        y0 = jtu.tree_map(lambda v, i, s: jnp.array([v, i, s]), self.v0, self.i0, s0)
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
            bm = diffrax.VirtualBrownianTree(t0, t1, tol=dt0 / 2, shape=bm_shape, key=bm_key)
            cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
            terms = diffrax.MultiTerm(vf, cvf)
        else:
            terms = vf

        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()

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
            event_types = jtu.tree_map(
                lambda et, em: et.at[state.num_spikes].set(em), event_types, event_mask
            )
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
            ytrans = jtu.tree_map(
                ft.partial(
                    _inner_trans_fn,
                    y=yevent,
                    event_mask=event_mask,
                    key=trans_key,
                    v_reset=self.v_reset,
                ),
                event_mask,
                self.w,
            )
            ytrans = get_switch(ytrans, event_idx, flatten_one=True)
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


def _build_forward_w(in_size, out_size, width_size, depth):
    if depth <= 1:
        width_size = out_size
    num_neurons = in_size + width_size * (depth - 1) + out_size
    neurons = tuple(i for i in range(num_neurons))
    children = tuple(i for i in range(in_size, in_size + width_size))
    w_in = tuple(tuple(True if i in children else False for i in neurons) for j in range(in_size))
    w_hidden = ()
    for n in range(1, depth):
        children = tuple(i for i in range(in_size + width_size * n, in_size + width_size * (n + 1)))
        w_hidden = (
            *w_hidden,
            *tuple(
                tuple(True if i in children else False for i in neurons) for j in range(width_size)
            ),
        )
    w_out = tuple(tuple(False for i in neurons) for j in range(out_size))
    w = (*w_in, *w_hidden, *w_out)
    return w


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
        w = _build_forward_w(self.in_size, self.out_size, self.width_size, self.depth)
        super().__init__(num_neurons, intensity_fn, w=w, key=key, diffusion=diffusion)

    def __call__(
        self,
        input_current: Callable[..., Float[Array, " input_size"]],
        ts: Float[Array, ""],
        max_spikes: int,
        key: Any,
        return_type: str = "spike_train",
    ):
        nn_minus_input = self.width_size * (self.depth - 1) + self.out_size
        nn_minus_output = nn_minus_input - self.out_size + self.in_size
        _input_current = tuple(
            lambda t, n=n: input_current(t)[n] for n in self.neurons[: self.in_size]
        )
        _input_current = (*_input_current, *tuple(lambda t: 0.0 for _ in range(nn_minus_input)))
        out = super().__call__(_input_current, ts, max_spikes, key=key)
        if return_type == "spike_train":
            spike_trains = jnp.array(jax.vmap(out.spike_train.evaluate)(ts)).T
            spike_trains = spike_trains[:, nn_minus_output:]
            return spike_trains
        elif return_type == "solution":
            return out
