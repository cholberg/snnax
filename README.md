# snnax

## Description

Spiking Neural Networks implemented on top of [diffrax](https://github.com/patrick-kidger/diffrax). Features include:

- simulating trajectories of Leaky-Integrate-and-Fire neurons;
- stochastic firing through intensity functions;
- Stochastic Spiking Neural Networks (SSNNs) as introduced [here](https://arxiv.org/abs/2405.13587);
- arbitrary network structures;
- automatic differentiation of spike times and neuronal state variables.

This project is still in a very early experimental phase. For now everything should work more or less out of the box, but I am not sure how much I will maintain this project in the future.

_One thing to note is that the latest relase of `jax` (v0.6.0) introduced a [bug](https://github.com/jax-ml/jax/pull/28150) that manifests in the current implementation. For now, installing this repository will also install v0.5.3, but I will probably update the dependencies in the future, when the fix is included._

## Installation

Clone the repository:

```
git clone https://github.com/cholberg/snnax
cd snnax
```

and install:

```
pip install .
```

## Usage

For some usage examples see the [example notebook](./notebooks/example.ipynb). To reproduce the results of [Exact Gradients for Stochastic Spiking Neural Networks Driven by Rough Signals](https://arxiv.org/abs/2405.13587) simply run the notebooks [here](./notebooks/single_neuron.ipynb) and [here](./notebooks/spiking_neural_net.ipynb).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
