import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp

from neural_actor import AbstractNeuralActor, AbstractOutputMapping, AbstractVectorField
from utils import mlp_init


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

        self.field = mlp_init(
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
        observations_size: int,
        width_size: int,
        depth: int,
        out_size: int,
        *,
        key: Array,
    ) -> None:
        self.map = mlp_init(
            in_size=state_shape + observations_size,
            width_size=width_size,
            depth=depth,
            out_size=out_size,
            key=key,
        )

    def __call__(self, y: Array, x: Array, *, key: Array | None = None) -> Array:
        return self.map(jnp.concatenate([y, x]))


class UnbiasedNeuralActor(AbstractNeuralActor[UnbiasedField, UnbiasedMap]):
    vector_field: UnbiasedField
    output_mapping: UnbiasedMap
    initial_state_mapping: eqx.nn.MLP

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
        input_key, output_key, initial_state_key = jax.random.split(key, 3)

        self.vector_field = UnbiasedField(
            state_shape,
            input_size,
            input_mapping_width,
            input_mapping_depth,
            key=input_key,
        )

        self.output_mapping = UnbiasedMap(
            state_shape,
            input_size,
            output_mapping_width,
            output_mapping_depth,
            output_size,
            key=output_key,
        )

        self.initial_state_mapping = eqx.nn.MLP(
            in_size=input_size,
            out_size=state_shape,
            width_size=input_mapping_width,
            depth=input_mapping_depth,
            key=initial_state_key,
        )

    def initial_state(self, x: Array | None = None, *, key: Array | None) -> Array:
        assert x is not None, "Initial state requires an input"
        return self.initial_state_mapping(x)
