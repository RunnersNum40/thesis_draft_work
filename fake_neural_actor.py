import equinox as eqx
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from utils import mlp_init


class FakeNeuralActor:
    mlp: eqx.nn.MLP
    state_shape: tuple[int]

    def __init__(
        self,
        in_size: int,
        width_size: int,
        depth: int,
        out_size: int,
        key: Array,
        *args,
        **kwargs,
    ) -> None:
        self.state_shape = (1,)
        self.mlp = mlp_init(
            in_size=in_size,
            width_size=width_size,
            depth=depth,
            out_size=out_size,
            final_std=0.01,
            key=key,
            *args,
            **kwargs,
        )

    def __call__(
        self,
        ts: Array,
        y0: Array,
        x: ArrayLike,
        *,
        map_output: bool = True,
        max_steps: int = 4096,
        adaptive_step_size: bool = False,
    ) -> tuple[Array, Array | None]:
        return y0, self.mlp(jnp.asarray(x))

    def initial_state(self, key: Array | None) -> Array:
        return jnp.zeros(self.state_shape)
