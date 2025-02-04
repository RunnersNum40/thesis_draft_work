import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


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
    ) -> None:
        super(CPG, self).__init__()

        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.timestep = timestep

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
            cpg_ode, state, torch.tensor([0, self.timestep]), method="heun3"
        )

        return solution[-1]

    def random_state(self) -> torch.Tensor:
        """
        Sample a random state for the CPG system.

        :return: A random state for the CPG system.
        """
        return states_to_tensor(
            torch.ones(self.num_oscillators),
            torch.zeros(self.num_oscillators),
            torch.zeros(self.num_oscillators),
        )


class CPGNetwork(nn.Module):
    """MLP network with a CPG layer"""

    def __init__(
        self,
        input_layers: list[int],
        num_oscillators: int,
        output_layers: list[int],
        timestep: float = 0.01,
        convergence_factor: float = 1.0,
    ) -> None:
        super(CPGNetwork, self).__init__()

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
                    for input_size, output_size in zip(input_layers, input_layers[1:])
                ]
            )
            if input_layers
            else nn.Identity()
        )

        self.cpg_layer = CPG(
            num_oscillators=num_oscillators,
            convergence_factor=convergence_factor,
            timestep=timestep,
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

    def sample_state(self) -> torch.Tensor:
        """
        Sample a random state for the CPG system.

        :return: A random state for the CPG system.
        """
        return self.cpg_layer.random_state()


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import TqdmExperimentalWarning
    from tqdm.rich import tqdm

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    def square(angle: float | np.ndarray, half_size: float) -> float | np.ndarray:
        x, y = np.cos(angle), np.sin(angle)

        if np.abs(x) > np.abs(y):
            x_bound = np.sign(x)
            y_bound = y / np.abs(x)
        else:
            y_bound = np.sign(y)
            x_bound = x / np.abs(y)

        return np.array([x_bound, y_bound]) * half_size

    time = 1.0
    timestep = 1e-2
    half_size = 1.0
    frequency = 1.0  # Hz
    t = np.linspace(0, time, int(time / timestep))
    y = np.array([square(angle, half_size) for angle in 2 * np.pi * frequency * t])

    t = torch.tensor(t, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    def visualize(model: CPGNetwork, t: torch.Tensor, y: torch.Tensor) -> None:
        model.eval()

        with torch.no_grad():
            state = model.sample_state()
            y_pred = torch.empty((len(t), 2))
            for i in range(len(t)):
                state, y_pred[i] = model(state, t[i])

        y_pred = y_pred.numpy()

        plt.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted")
        plt.plot(y[:, 0], y[:, 1], label="True")
        plt.legend()
        plt.show()

        plt.plot(t, y_pred[:, 0], "r-", label="Predicted x")
        plt.plot(t, y[:, 0], "r--", label="True x")
        plt.plot(t, y_pred[:, 1], "b-", label="Predicted y")
        plt.plot(t, y[:, 1], "b--", label="True y")
        plt.legend()
        plt.show()

    def train(
        model: CPGNetwork,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1000,
        lr: float = 1e-3,
        report_frequency: int = 100,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in tqdm(range(epochs), leave=False):
            optimizer.zero_grad()
            loss = torch.tensor(0.0)

            state = model.sample_state()
            for i in range(len(y)):
                state, y_pred = model(state, x[i])
                loss += criterion(y_pred, y[i])

            loss /= len(y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if epoch % report_frequency == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        print(f"Epoch {epochs}/{epochs}, Loss: {loss.item()}")

    n = 3
    model = CPGNetwork(
        [1, 16, 16],
        n,
        [16, 16, 2],
        timestep=timestep,
        convergence_factor=10.0,
    )

    train(model, t, y, 500)
    visualize(model, t, y)
