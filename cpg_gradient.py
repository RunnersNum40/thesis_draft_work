from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint


def states_to_tensor(
    amplitudes: torch.Tensor, phases: torch.Tensor, amplitudes_dot: torch.Tensor
) -> torch.Tensor:
    return torch.cat([amplitudes, phases, amplitudes_dot])


def tensor_to_states(
    state: torch.Tensor, num_oscillators: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amplitudes = state[:num_oscillators]
    amplitudes_dot = state[num_oscillators * 2 : 3 * num_oscillators]
    phases = state[num_oscillators : 2 * num_oscillators]

    return amplitudes, phases, amplitudes_dot


class CPGOde(nn.Module):
    """ODE Function for use with odeint"""

    def __init__(
        self,
        convergence_factor: float,
        intrinsic_amplitudes,
        intrinsic_frequencies,
        coupling_weights,
        phase_biases,
    ) -> None:
        super(CPGOde, self).__init__()

        self.convergence_factor = convergence_factor
        self.intrinsic_amplitudes = intrinsic_amplitudes
        self.intrinsic_frequencies = intrinsic_frequencies
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases

        self.num_oscillators = len(intrinsic_amplitudes)

    def forward(self, t, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CPG ODE.

        :param t: The current time.
        :param y: The current state of the CPG system.
        :return: The derivative of the state of the CPG system.
        """
        amplitudes, amplitudes_dot, phases = tensor_to_states(y, self.num_oscillators)

        phase_dots = self.intrinsic_frequencies + torch.sum(
            (
                amplitudes[:, None]
                * self.coupling_weights
                * torch.sin(phases[None, :] - phases[:, None] - self.phase_biases)
            ),
            dim=1,
        )

        amplitudes_dot_dot = self.convergence_factor * (
            self.convergence_factor / 4 * (self.intrinsic_amplitudes - amplitudes)
            - amplitudes_dot
        )

        return states_to_tensor(amplitudes_dot, phase_dots, amplitudes_dot_dot)


class CPG(nn.Module):
    """NN layer of the CPG model"""

    def __init__(
        self,
        num_oscillators: int,
        convergence_factor: float,
        solver: Optional[str],
    ) -> None:
        super(CPG, self).__init__()

        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.solver = solver

        num_params = (
            num_oscillators  # intrinsic amplitudes
            + num_oscillators  # intrinsic frequencies
            + num_oscillators**2  # coupling weights
            + num_oscillators**2  # phase biases
        )
        self.input_size = num_params

    def forward(
        self, state: torch.Tensor, params: torch.Tensor, timestep: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the CPG model.

        :param state: The current state of the CPG system.
        :param params: The parameters of the CPG system.
        :return: The next state of the CPG system.
        """
        intrinsic_amplitudes = params[: self.num_oscillators]
        intrinsic_frequencies = params[self.num_oscillators : 2 * self.num_oscillators]
        coupling_weights = params[
            2 * self.num_oscillators : 2 * self.num_oscillators
            + self.num_oscillators**2
        ].reshape(self.num_oscillators, self.num_oscillators)
        phase_biases = params[
            2 * self.num_oscillators + self.num_oscillators**2 :
        ].reshape(self.num_oscillators, self.num_oscillators)

        cpg_ode = CPGOde(
            intrinsic_amplitudes=intrinsic_amplitudes,
            intrinsic_frequencies=intrinsic_frequencies,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_factor=self.convergence_factor,
        )

        solution = odeint(
            cpg_ode, state, torch.tensor([0, timestep]), method=self.solver
        )

        if solution is None:
            raise ValueError("Integration failed")

        return solution[-1]  # pyright: ignore


class CPGNetwork(nn.Module):
    """MLP network with a CPG layer

    A CPG network contains an input processing network, a stateful CPG layer, and an
    output processing network. The input network determines the parameters of the CPG,
    and the output network determines an output from the CPG state.
    """

    def __init__(
        self,
        input_layers: list[int],
        num_oscillators: int,
        output_layers: list[int],
        convergence_factor: float = 1000.0,
        solver: Optional[str] = "rk4",
        state_feedback: bool = False,
        param_feedback: bool = False,
        bypass_layer: int = 0,
    ) -> None:
        """
        Create a CPG network.

        :param input_layers: The sizes of the layers in the input network.
            If the list is empty, the input size must match the number of oscillator parameters.
        :param num_oscillators: The number of oscillators in the CPG system.
        :param output_layers: The sizes of the layers in the output network.
            If the list is empty, the number of oscillators must match the output size // 2.
        :param timestep: The timestep for the CPG system.
        :param convergence_factor: The convergence factor for the CPG system.
        :param solver: The solver to use for the CPG system.
        :param state_feedback: Whether to feed the CPG state into the input processing network.
        :param param_feedback: Whether to feed the computed CPG parameters into the output
            processing network.
        """

        super(CPGNetwork, self).__init__()

        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.solver = solver
        self.param_feedback = param_feedback
        self.state_feedback = state_feedback
        self.bypass_layer = bypass_layer

        self.state_shape = (3 * self.num_oscillators,)

        input_final_layer_size = (
            num_oscillators  # intrinsic amplitudes
            + num_oscillators  # intrinsic frequencies
            + num_oscillators**2  # coupling weights
            + num_oscillators**2  # phase biases
        )
        input_layers[0] += 1  # timestep
        if state_feedback:
            input_layers[0] += 2 * num_oscillators
        input_layers = input_layers + [input_final_layer_size + bypass_layer]
        self.input_network = self.make_subnetwork(input_layers)

        self.cpg_layer = CPG(
            num_oscillators=num_oscillators,
            convergence_factor=convergence_factor,
            solver=solver,
        )

        out_first_layer_size = 2 * num_oscillators
        if param_feedback:
            out_first_layer_size += input_final_layer_size

        output_layers = [out_first_layer_size + bypass_layer] + output_layers
        self.output_network = self.make_subnetwork(output_layers)

    @staticmethod
    def make_subnetwork(layers: list[int]) -> nn.Sequential | nn.Identity | nn.Linear:
        """
        Return a subnetwork based on the given layers.

        A subnetwork is a sequence of linear layers with ReLU activation functions in between
        all layers and not after last one.

        :param layers: The sizes of the layers in the subnetwork.
        "return: A subnetwork based on the given layers.
        """
        if len(layers) == 1:
            return nn.Identity()

        if len(layers) == 2:
            return nn.Linear(layers[0], layers[1])

        layers_ = []
        for n, (in_size, out_size) in enumerate(zip(layers, layers[1:])):
            layers_.append(nn.Linear(in_size, out_size))
            if n < len(layers) - 2:
                layers_.append(nn.ReLU())

        return nn.Sequential(*layers_)

    def forward(
        self, state: torch.Tensor, x: torch.Tensor, timestep: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CPG network.

        :param state: The current state of the CPG system.
        :param x: The input to the network.
        :return: The output of the network.
        """
        if x.ndim < 1:
            x = x.unsqueeze(0)

        if self.state_feedback:
            amplitudes, phases, _ = tensor_to_states(state, self.num_oscillators)
            x = torch.cat(
                [
                    x,
                    amplitudes,
                    phases,
                ]
            )

        x = torch.cat([x, torch.tensor([timestep])])

        x = self.input_network(x)

        if self.bypass_layer:
            params, bypass = x[: -self.bypass_layer], x[-self.bypass_layer :]
        else:
            params, bypass = x, torch.tensor([])

        state = self.cpg_layer(state, params, timestep)

        amplitudes = state[: self.cpg_layer.num_oscillators]
        phases = state[
            self.cpg_layer.num_oscillators : 2 * self.cpg_layer.num_oscillators
        ]
        cpg_output = torch.cat(
            [amplitudes * torch.cos(phases), amplitudes * torch.sin(phases)]
        )

        if self.param_feedback:
            x = torch.cat([params, cpg_output, bypass])
        else:
            x = torch.cat([cpg_output, bypass])

        x = self.output_network(x)

        return state.detach(), x


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import TqdmExperimentalWarning
    from tqdm.rich import trange

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    def square(angle: np.ndarray, half_size: float) -> np.ndarray:
        """
        Returns the position of points on a square with side length 2 * half_size for given angles.

        :param angle: Array of angles from the x-axis.
        :param half_size: Half the side length of the square.
        :return: The positions of the points as an array of shape (N, 2).
        """
        x, y = np.cos(angle), np.sin(angle)

        x_bound = np.zeros_like(x)
        y_bound = np.zeros_like(y)

        mask_x = np.abs(x) > np.abs(y)

        x_bound[mask_x] = np.sign(x[mask_x])
        y_bound[mask_x] = y[mask_x] / np.abs(x[mask_x])

        x_bound[~mask_x] = x[~mask_x] / np.abs(y[~mask_x])
        y_bound[~mask_x] = np.sign(y[~mask_x])

        return np.stack([x_bound, y_bound], axis=-1) * half_size

    def default_state(num_oscillators: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.ones(num_oscillators),
                torch.zeros(num_oscillators),
                torch.zeros(num_oscillators),
            ]
        )

    def visualize(
        model: CPGNetwork,
        t: torch.Tensor,
        y: torch.Tensor,
        timestep: float | torch.Tensor,
    ) -> None:
        model.eval()

        with torch.no_grad():
            state = default_state(model.cpg_layer.num_oscillators)
            amplitudes = []
            phases = []
            y_pred = torch.empty((len(t), 2))
            for i in range(len(t)):
                state, y_pred[i] = model(state, t[i], timestep)
                amplitude, phase, _ = tensor_to_states(
                    state, model.cpg_layer.num_oscillators
                )
                amplitudes.append(amplitude)
                phases.append(phase)

        y_pred = y_pred.numpy()
        amplitudes = torch.stack(amplitudes).numpy()
        phases = torch.stack(phases).numpy()

        occilator_x = amplitudes * np.cos(phases)
        occilator_y = amplitudes * np.sin(phases)

        plt.plot(y[:, 0], y[:, 1], "b--", label="True")
        plt.plot(y_pred[:, 0], y_pred[:, 1], "r-", label="Predicted")
        plt.legend()
        plt.show()

        plt.plot(y[:, 0], y[:, 1], "b--", label="True")
        plt.plot(occilator_x, occilator_y, "-", label="Oscillators")
        plt.legend()
        plt.show()

        plt.plot(t, y[:, 0], "b--", label="True x")
        plt.plot(t, y[:, 1], "r--", label="True y")
        plt.plot(t, y_pred[:, 0], "b-", label="Predicted x")
        plt.plot(t, y_pred[:, 1], "r-", label="Predicted y")
        plt.legend()
        plt.show()

        plt.plot(t, y[:, 0], "b--", label="True x")
        plt.plot(t, y[:, 1], "r--", label="True y")
        plt.plot(t, occilator_x, "-", label="Oscillator x")
        plt.plot(t, occilator_y, "-", label="Oscillator y")

        plt.legend()
        plt.show()

    def train(
        model: CPGNetwork,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1000,
        lr: float = 1e-4,
        report_frequency: int = 100,
        timestep: float = 5e-3,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in trange(epochs, leave=False):
            optimizer.zero_grad()
            loss = torch.tensor(0.0)

            state = default_state(model.cpg_layer.num_oscillators)
            for i in range(len(y)):
                state, y_pred = model(state, x[i], timestep)
                loss += criterion(y_pred, y[i])

            loss /= len(y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if (epoch + 1) % report_frequency == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1 if epoch > 0 else 0}/{epochs}, Loss: {loss.item()}"
                )

    time = 2.0
    timestep = 5e-3
    half_size = 1.0
    frequency = 1.0  # Hz
    t = np.linspace(0, time, int(time / timestep))
    y = square(2 * np.pi * frequency * t, half_size)

    t = torch.tensor(t, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n = 2
    model = CPGNetwork(
        [1, 64, 64],
        n,
        [2],
        convergence_factor=1000.0,
        solver="rk4",
        state_feedback=True,
    )

    train(model, t, y, epochs=3000, lr=1e-4, timestep=timestep)
    visualize(model, t, y, timestep=timestep)
