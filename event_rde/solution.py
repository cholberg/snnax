import equinox as eqx
from jaxtyping import Array, PyTree

from .paths import SpikeTrain


class Solution(eqx.Module):
    ys: PyTree[Array, " neuron"]
    ts: Array
    spike_times: Array
    spike_marks: PyTree[Array, " neuron"]
    spike_values: PyTree[Array, " neuron"]
    spike_train: SpikeTrain
    num_spikes: int
    max_spikes: int
