from typing import Optional

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from .paths import SpikeTrain


class Solution(eqx.Module):
    ys: Optional[PyTree[Array, " neuron"]]
    ts: Array
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neuron"]
    spike_values: Float[Array, "samples spikes neuron 3"]
    spike_train: SpikeTrain
    num_spikes: int
    max_spikes: int
