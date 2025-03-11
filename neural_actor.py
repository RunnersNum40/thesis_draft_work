from abc import abstractmethod
from typing import Generic, TypeVar

import diffrax
import equinox as eqx
import jax
from jax import Array
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


VF = TypeVar("VF", bound="AbstractVectorField")
OM = TypeVar("OM", bound="AbstractOutputMapping")


class AbstractVectorField(eqx.Module, strict=True):
    """Abstract module to compute the vector field of a dynamical system."""

    state_shape: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, t: float, y: Array, x: Array) -> Array:
        """Compute the vector field at a given state with external input."""
        raise NotImplementedError


class AbstractOutputMapping(eqx.Module, strict=True):
    """Abstract module to map a system's state to an actor output."""

    @abstractmethod
    def __call__(self, y: Array) -> Array:
        """Map a state to an actor output."""
        raise NotImplementedError


class AbstractNeuralActor(eqx.Module, Generic[VF, OM], strict=True):
    """Abstract module to represent a neural actor model.

    Neural actor models are dynamical systems with external input and output mapping.
    """

    vector_field: eqx.AbstractVar[VF]
    output_mapping: eqx.AbstractVar[OM]

    def __check_init__(self):
        if not isinstance(self.vector_field, AbstractVectorField):
            raise TypeError(
                f"Expected vector_field to be a subclass of AbstractVectorField, got {self.vector_field} instead"
            )

        if not isinstance(self.output_mapping, AbstractOutputMapping):
            raise TypeError(
                f"Expected output_mapping to be a subclass of AbstractOutputMapping, got {self.output_mapping} instead"
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
        term = diffrax.ODETerm(self.vector_field)  # pyright: ignore
        solver = diffrax.Heun()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0]
        saveat = diffrax.SaveAt(t1=True)
        if adaptive_step_size:
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            args=x,
            max_steps=max_steps,
        )

        assert solution.ys is not None
        state = solution.ys[-1]

        if map_output:
            return state, self.output_mapping(state)

        return state, None

    @property
    def state_shape(self) -> tuple[int]:
        return (self.vector_field.state_shape,)
