from typing import Literal

import equinox as eqx
from jax import Array
from jax import random as jr


def mlp_init(
    in_size: int | Literal["scalar"],
    width_size: int,
    depth: int,
    out_size: int | Literal["scalar"],
    key: Array,
    hidden_std: float | None = None,
    final_std: float | None = None,
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

    if hidden_std is not None:
        for i in range(len(mlp.layers) - 1):
            new_weights = (
                jr.normal(weights_key, mlp.layers[i].weight.shape) * hidden_std
            )
            mlp = eqx.tree_at(lambda m: m.layers[i].weight, mlp, new_weights)

    if final_std is not None:
        new_weights = jr.normal(weights_key, mlp.layers[-1].weight.shape) * final_std
        mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp, new_weights)
    return mlp
