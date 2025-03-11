import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp

from neural_actor import AbstractNeuralActor, AbstractOutputMapping, AbstractVectorField


class UnbiasedField(AbstractVectorField):
    state_shape: int
    field: eqx.nn.MLP

    def __init__(
        self,
        state_shape: int,
        in_size: int,
        width_size: int,
        depth: int,
        *,
        key: Array,
    ) -> None:
        self.state_shape = state_shape

        self.field = eqx.nn.MLP(
            in_size=in_size + 1 + state_shape,
            out_size=state_shape,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.softplus,  # Continuously differentiable activation function theoretically required
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t: float, y: Array, x: Array) -> Array:
        return self.field(jnp.concatenate([jnp.asarray([t]), y, x]))


class UnbiasedMap(AbstractOutputMapping):
    map: eqx.nn.MLP

    def __init__(
        self,
        state_shape: int,
        width_size: int,
        depth: int,
        out_size: int,
        *,
        key: Array,
    ) -> None:
        self.map = eqx.nn.MLP(
            in_size=state_shape,
            width_size=width_size,
            depth=depth,
            out_size=out_size,
            key=key,
        )

    def __call__(self, y: Array) -> Array:
        return self.map(y)


class UnbiasedNeuralActor(AbstractNeuralActor[UnbiasedField, UnbiasedMap]):
    vector_field: UnbiasedField
    output_mapping: UnbiasedMap

    def __init__(
        self,
        state_shape: int,
        input_size: int,
        input_mapping_width: int,
        input_mapping_depth: int,
        output_size: int,
        output_mapping_width: int,
        output_mapping_depth: int,
        *,
        key: Array,
    ) -> None:
        input_key, output_key = jax.random.split(key)

        self.vector_field = UnbiasedField(
            state_shape,
            input_size,
            input_mapping_width,
            input_mapping_depth,
            key=input_key,
        )

        self.output_mapping = UnbiasedMap(
            state_shape,
            output_mapping_width,
            output_mapping_depth,
            output_size,
            key=output_key,
        )
