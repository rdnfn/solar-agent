"""This module contains photovoltaic system models."""

import numpy as np


class PhotovoltaicModel:
    """Base class for photovoltaic (PV) system models."""

    def __init__(self) -> None:
        """Base class for photovoltaic (PV) system models."""
        pass

    def get_next_generation(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """

    def get_prediction(self, start_time: float, end_time: float) -> np.array:
        """Get prediction of future PV generation.

        Args:
            start_time (float): begin of prediction
            end_time (float): end of prediction

        Returns:
            np.array: predicted generation (kW)
        """
        raise NotImplementedError


class DataPV(PhotovoltaicModel):
    """Photovoltaic model that samples from data."""

    def __init__(
        self, data_path: str, time_step_len: float = 1, num_steps: int = 24
    ) -> None:
        """Photovoltaic model that samples from data.

        Args:
            data_path (str): path to PV data in a txt file with solar trace in kW
            time_step_len (float): length of time steps in hours. Defaults to 1.
            num_steps (int): number of time steps. Defaults to 24.
        """
        super().__init__()

        self.data = np.loadtxt(data_path, delimiter=",")
        self.num_steps = num_steps
        self.time_step_len = time_step_len

        self.reset()

    def reset(self, start: int = None) -> None:
        """Reset the PV system to new randomly sampled data."""

        self.time_step = 0

        # Set values for entire episode
        if start is None:
            start = np.random.randint(len(self.data) // 24) * 24
        self.start = start
        end = start + self.num_steps
        self.episode_values = self.data[start:end]

    def step(self) -> None:
        self.time_step += 1

    def get_next_generation(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """
        gen_power = self.episode_values[self.time_step]
        self.step()
        return gen_power

    def get_prediction(self, start_time: float, end_time: float) -> np.array:
        """Get prediction of future PV generation.

        Args:
            start_time (float): begin of prediction
            end_time (float): end of prediction

        Returns:
            np.array: predicted generation (kW)
        """
        return self.episode_values[start_time:end_time]
