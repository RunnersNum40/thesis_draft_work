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
        timestep: float = 0.01,
        solver: Optional[str] = "rk4",
    ) -> None:
        super(CPG, self).__init__()

        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.timestep = timestep
        self.solver = solver

        num_params = (
            num_oscillators  # intrinsic amplitudes
            + num_oscillators  # intrinsic frequencies
            + num_oscillators**2  # coupling weights
            + num_oscillators**2  # phase biases
        )
        self.input_size = num_params

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
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
            cpg_ode, state, torch.tensor([0, self.timestep]), method=self.solver
        )

        return solution[-1]


class CPGNetwork(nn.Module):
    """MLP network with a CPG layer"""

    def __init__(
        self,
        input_layers: list[int],
        num_oscillators: int,
        output_layers: list[int],
        timestep: float = 0.01,
        convergence_factor: float = 1.0,
        solver: Optional[str] = "rk4",
    ) -> None:
        super(CPGNetwork, self).__init__()

        self.num_oscillators = num_oscillators
        self.timestep = timestep
        self.convergence_factor = convergence_factor
        self.solver = solver

        input_final_layer_size = (
            num_oscillators  # intrinsic amplitudes
            + num_oscillators  # intrinsic frequencies
            + num_oscillators**2  # coupling weights
            + num_oscillators**2  # phase biases
        )
        input_layers = input_layers + [input_final_layer_size]
        self.input_network = (
            nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
                    for input_size, output_size in zip(
                        input_layers[:-1], input_layers[:-1][1:]
                    )
                ]
            )
            if input_layers
            else nn.Identity()
        )
        self.input_network.add_module(
            "final", nn.Linear(input_layers[-2], input_layers[-1])
        )

        self.cpg_layer = CPG(
            num_oscillators=num_oscillators,
            convergence_factor=convergence_factor,
            timestep=timestep,
            solver=solver,
        )

        out_first_layer_size = 2 * num_oscillators
        output_layers = [out_first_layer_size] + output_layers
        self.output_network = (
            nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
                    for input_size, output_size in zip(
                        output_layers[:-1], output_layers[:-1][1:]
                    )
                ]
            )
            if output_layers
            else nn.Identity()
        )
        self.output_network.add_module(
            "final", nn.Linear(output_layers[-2], output_layers[-1])
        )

    def forward(
        self, state: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CPG network.

        :param state: The current state of the CPG system.
        :param x: The input to the network.
        :return: The output of the network.
        """
        if x.ndim < 1:
            x = x.unsqueeze(0)

        x = self.input_network(x)

        state = self.cpg_layer(state, x)

        amplitudes = state[: self.cpg_layer.num_oscillators]
        phases = state[
            self.cpg_layer.num_oscillators : 2 * self.cpg_layer.num_oscillators
        ]
        x = torch.cat([amplitudes * torch.sin(phases), amplitudes * torch.cos(phases)])

        x = self.output_network(x)

        return state.detach(), x


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import TqdmExperimentalWarning
    from tqdm.rich import trange

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    def square(angle: float | np.ndarray, half_size: float) -> np.ndarray:
        """
        Returns the position of a point on a square with side length 2 * half_size for a given angle.

        :param angle: The angle of the point from the x-axis.
        :param half_size: Half the side length of the square.
        :return: The position of the point.
        """
        x, y = np.cos(angle), np.sin(angle)

        x_bound, y_bound = np.zeros_like(x), np.zeros_like(y)

        if np.abs(x) > np.abs(y):
            y_bound = y / np.abs(x)
            x_bound = np.sign(x)
        else:
            x_bound = x / np.abs(y)
            y_bound = np.sign(y)

        return np.stack([x_bound, y_bound], axis=-1) * half_size

    time = 1.0
    timestep = 5e-3
    half_size = 1.0
    frequency = 1.0  # Hz
    t = np.linspace(0, time, int(time / timestep))
    y = np.array([square(2 * np.pi * frequency * t, half_size) for t in t])

    t = torch.tensor(t, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    def default_state(num_oscillators: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.ones(num_oscillators),
                torch.zeros(num_oscillators),
                torch.zeros(num_oscillators),
            ]
        )

    def visualize(model: CPGNetwork, t: torch.Tensor, y: torch.Tensor) -> None:
        model.eval()

        with torch.no_grad():
            state = default_state(model.cpg_layer.num_oscillators)
            amplitudes = []
            phases = []
            y_pred = torch.empty((len(t), 2))
            for i in range(len(t)):
                state, y_pred[i] = model(state, t[i])
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
        plt.plot(t, y_pred[:, 0], "r-", label="Predicted x")
        plt.plot(t, y_pred[:, 1], "b-", label="Predicted y")
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
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in trange(epochs, leave=False):
            optimizer.zero_grad()
            loss = torch.tensor(0.0)

            state = default_state(model.cpg_layer.num_oscillators)
            for i in range(len(y)):
                state, y_pred = model(state, x[i])
                loss += criterion(y_pred, y[i])

            loss /= len(y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if (epoch + 1) % report_frequency == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1 if epoch > 0 else 0}/{epochs}, Loss: {loss.item()}"
                )

    n = 4
    model = CPGNetwork(
        [1, 64, 64],
        n,
        [64, 64, 2],
        timestep=timestep,
        convergence_factor=1000.0,
        solver="euler",
    )

    train(model, t, y, 10000, lr=1e-5)
    visualize(model, t, y)
