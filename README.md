# snnax

## Description
Spiking Neural Networks implemented on top of [diffrax](https://github.com/patrick-kidger/diffrax). Features include:

- simulating trajectories of Leaky-Integrate-and-Fire neurons;
- stochastic firing through intensity functions;
- Stochastic Spiking Neural Networks (SSNNs) as introduced [here](https://arxiv.org/abs/2405.13587);
- arbitrary network structures;
- automatic differentiation of spike times and neuronal state variables.

This project is still in a very early experimental phase. Future features might include more complex neuronal dynamics, simulation of exact solutions, and custom gradients for online learning.

## Installation

Clone the repository:

```
git clone https://github.com/cholberg/snnax
cd snnax
pip install .
```

Make sure to have jax installed. For CPU:

```
pip install "jax[cpu]"
```

For NVIDIA GPU:

```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

For some usage examples see the [example notebook](./notebooks/example.ipynb). To reproduce the results of [Exact Gradients for Stochastic Spiking Neural Networks Driven by Rough Signals](https://arxiv.org/abs/2405.13587) simply run the notebooks [here](./notebooks/single_neuron.ipynb) and [here](./notebooks/spiking_neural_net.ipynb).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
