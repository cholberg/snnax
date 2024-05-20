from typing import Literal, Optional, Tuple, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractPath, BrownianIncrement, SpaceTimeLevyArea, VirtualBrownianTree
from diffrax._brownian.tree import _levy_diff, _make_levy_val
from diffrax._custom_types import RealScalarLike, levy_tree_transpose
from diffrax._misc import linear_rescale
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Real
from typing_extensions import TypeAlias

_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]


def _plottable_neuron(ts, ys):
    _, times, _ = ys.shape
    idx = ys[:, :, 2] < 0
    ys_flat = ys[idx]
    ts_flat = ts[idx]
    ts_out = jnp.linspace(ts_flat[0], ts_flat[-1], times)
    return ys_flat[jnp.searchsorted(ts_flat, ts_out), :]


def plottable_path(
    ts: Real[Array, "spikes times"], ys: Float[Array, "spikes neurons times 3"]
) -> Tuple[Real[Array, " times"], Float[Array, "neurons times 3"]]:
    t0, t1 = ts[0, [0, -1]]
    _, neurons, times, _ = ys.shape
    out_path = []
    for n in range(neurons):
        out_path.append(_plottable_neuron(ts, ys[:, n]))
    ts = jnp.linspace(t0, t1, times)
    ys = jnp.transpose(jnp.dstack(out_path), (2, 0, 1))
    return ts, ys


def interleave(arr1: Array, arr2: Array) -> Array:
    out = jnp.empty((arr1.size + arr2.size,), dtype=arr1.dtype)
    out = out.at[0::2].set(arr2)
    out = out.at[1::2].set(arr1)
    return out


def marcus_lift(
    t0: RealScalarLike,
    t1: RealScalarLike,
    spike_times: Float[Array, " max_spikes"],
    spike_mask: Float[Array, "max_spikes num_neurons"],
) -> Float[Array, " 2_max_spikes"]:
    """Lifts a spike train to a path to a discretisation of the Marcus lift
    (with time augmentation).

    **Arguments**:

    - `t0`: The start time of the path.
    - `t1`: The end time of the path.
    - `spike_times`: The times of the spikes.
    - `spike_mask`: A mask indicating the corresponding spiking neuron.

    **Returns**:

    - An array of shape `(2 * max_spikes, num_neurons + 1)` representing the Marcus lift.
    """
    num_neurons = spike_mask.shape[1]
    finite_spikes = jnp.where(jnp.isfinite(spike_times), spike_times, t1).reshape((-1, 1))
    spike_cumsum = jnp.cumsum(spike_mask, axis=0)
    # last_spike_time = jnp.max(jnp.where(spike_times < t1, spike_times, -jnp.inf))
    spike_cumsum_shift = jnp.roll(spike_cumsum, 1, axis=0)
    spike_cumsum_shift = spike_cumsum_shift.at[0, :].set(
        jnp.zeros(num_neurons, dtype=spike_cumsum_shift.dtype)
    )
    arr1 = jnp.hstack([finite_spikes, spike_cumsum])
    arr2 = jnp.hstack([finite_spikes, spike_cumsum_shift])
    out = jax.vmap(interleave, in_axes=1)(arr1, arr2).T
    # Makes sure the path starts at t0
    out = jnp.roll(out, 1, axis=0)
    out = out.at[0, :].set(jnp.insert(jnp.zeros(num_neurons), 0, t0))
    # time_capped = jnp.where(out[:, 0] < t1, out[:, 0], last_spike_time)
    # out = out.at[:, 0].set(time_capped)
    return out


@eqx.filter_jit
def cap_fill_ravel(ts, ys, spike_cap=10):
    # Cap the number of spikes
    ys_capped = ys[:spike_cap]
    ts_capped = ts[:spike_cap]
    spikes, neurons, times, _ = ys_capped.shape

    # Fill up infs
    idx = ts_capped > ts_capped[:, -1, None]
    idx_y = jnp.tile(idx[:, None, :, None], (1, neurons, 1, 3))
    ts_capped = jnp.where(idx, ts_capped[:, -1, None], ts_capped)
    ys_capped = jnp.where(idx_y, ys_capped[:, :, -1, None, :], ys_capped)

    xs = (ts_capped, ys_capped)
    carry_ys = jnp.zeros((neurons, times, 3))
    carr_ts = jnp.array(0.0)
    carry = (carr_ts, carry_ys)

    def _fill(carry, x):
        _ts, _ys = x
        carry_ts, carry_ys = carry
        _ys_fill_val = jnp.tile(_ys[:, None, -1], (1, times, 1))
        _ts_fill_val = _ts[-1]
        _ys_out = jnp.where(jnp.isinf(_ys), _ys_fill_val, _ys)
        _ts_out = jnp.where(jnp.isinf(_ts), _ts_fill_val, _ts)
        assert isinstance(_ys_out, Array)
        _ys_all_inf = jnp.all(jnp.isinf(_ys_out))
        _ts_all_inf = jnp.all(jnp.isinf(_ts_out))
        _ys_out = jnp.where(_ys_all_inf, carry_ys, _ys_out)
        _ts_out = jnp.where(_ts_all_inf, carry_ts, _ts_out)
        assert isinstance(_ys_out, Array)
        new_carry_ys = jnp.tile(_ys_out[:, None, -1], (1, times, 1))
        new_carry_ts = _ts_out[-1]
        new_carry = (new_carry_ts, new_carry_ys)
        out = (_ts_out, _ys_out)
        return new_carry, out

    _, xs_filled_capped = jax.lax.scan(_fill, carry, xs=xs)
    ts_filled_capped, ys_filled_capped = xs_filled_capped

    # Ravel out the "spikes" dimension
    # (spikes, neurons, times, 3) -> (neurons, spikes, times, 3)
    ys_filled_capped = jnp.transpose(ys_filled_capped, (1, 0, 2, 3))
    # (spikes, neurons, times, 3) -> (neurons, spikes*times, 3)
    ys_filled_capped_ravelled = ys_filled_capped.reshape((neurons, -1, 3))
    ts_filled_capped_ravelled = jnp.ravel(ts_filled_capped)
    return ts_filled_capped_ravelled, ys_filled_capped_ravelled


class SpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    num_spikes: int
    spike_times: Array
    spike_cumsum: Array
    num_neurons: int

    def __init__(self, t0, t1, spike_times, spike_mask):
        max_spikes, num_neurons = spike_mask.shape
        self.num_neurons = num_neurons
        self.t0 = t0
        self.t1 = t1
        self.num_spikes = spike_times.shape[0]
        self.spike_times = jnp.insert(spike_times, 0, t0)
        self.spike_cumsum = jnp.cumsum(
            jnp.insert(spike_mask, 0, jnp.full_like(spike_mask[0], False), axis=0), axis=0
        )

    def evaluate(self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        idx = jnp.searchsorted(self.spike_times, t0)
        idx = jnp.where(idx > 0, idx - 1, idx)
        out = jax.lax.dynamic_slice(self.spike_cumsum, (idx, 0), (self.num_neurons, 1))[:, 0]
        return out


# A version of VirtualBrownianTree that will not throw an error when differentiated
class BrownianPath(VirtualBrownianTree):
    @eqxi.doc_remove_args("_spline")
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[Union[BrownianIncrement, SpaceTimeLevyArea]] = BrownianIncrement,
        _spline: _Spline = "sqrt",
    ):
        super().__init__(t0, t1, tol, shape, key, levy_area, _spline)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], BrownianIncrement, SpaceTimeLevyArea]:
        t0 = jax.lax.stop_gradient(t0)
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0
            levy_out = jtu.tree_map(_make_levy_val, self.shape, levy_out)
        else:
            t1 = jax.lax.stop_gradient(t1)
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, (BrownianIncrement, SpaceTimeLevyArea))
        return levy_out if use_levy else levy_out.W
