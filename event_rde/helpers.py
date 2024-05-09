import jax.numpy as jnp
import jax.random as jr
import numpy as np


def generate_weights(in_size, out_size, width_size, depth, key, w_sum=10.0):
    num_neurons = in_size + width_size * (depth - 1) + out_size
    w = jnp.zeros((num_neurons, num_neurons))
    layer_list = np.array([0] + [in_size] + [width_size] * (depth - 1) + [out_size])
    layer_idx = np.cumsum(np.array(layer_list))
    for i in range(depth):
        w_key = jr.fold_in(key, i)
        lrows = layer_idx[i]
        urows = layer_idx[i + 1]
        lcols = layer_idx[i + 1]
        ucols = layer_idx[i + 2]
        layer_size = layer_list[i + 1]
        w_fill = jr.uniform(w_key, (urows - lrows, ucols - lcols), minval=0.5, maxval=1.5)
        w_fill = w_fill * (w_sum / layer_size)
        w = w.at[lrows:urows, lcols:ucols].set(w_fill)
    return w
