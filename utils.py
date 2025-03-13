import equinox as eqx
from jax import Array
from jax import random as jr


def mlp_with_final_layer_std(
    in_size: int,
    width_size: int,
    depth: int,
    out_size: int,
    std: float,
    key: Array,
    *args,
    **kwargs,
) -> eqx.nn.MLP:
    kwargs = {}

    mlp_key, weights_key = jr.split(key)
    mlp = eqx.nn.MLP(
        in_size=in_size,
        width_size=width_size,
        depth=depth,
        out_size=out_size,
        key=mlp_key,
        *args,
        **kwargs,
    )
    new_weights = jr.normal(weights_key, mlp.layers[-1].weight.shape) * std
    mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp, new_weights)
    return mlp
