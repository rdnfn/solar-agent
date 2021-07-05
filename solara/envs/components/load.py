"""This module contains residential load models."""

import numpy as np

from solara.envs.components.base import EnvComponent


class LoadModel(EnvComponent):
    """Base class for residential load models."""

    def get_next_load(self) -> float:
        """Get power load for the next time step.

        Returns:
            float: power consumed (kW)
        """
        raise NotImplementedError


class DataLoad(LoadModel):
    """Model that samples load traces from data."""

    def __init__(
        self,
        data_path: str,
        time_step_len: float = 1,
        num_steps: int = 24,
        fixed_sample_num: int = None,
    ) -> None:
        """Load model that samples from data.

        Args:
            data_path (str): path to load data in a txt file with solar trace in kW
            time_step_len (float): length of time steps in hours. Defaults to 1.
            num_steps (int): number of time steps. Defaults to 24.
        """
        super().__init__()

        self.data = np.loadtxt(data_path, delimiter=",")
        self.num_steps = num_steps
        self.time_step_len = time_step_len
        self.fix_start(fixed_sample_num)

        self.reset()

    def reset(self, start: int = None) -> None:
        """Reset the load model to new randomly sampled data."""

        self.time_step = 0

        # Set values for entire episode
        if self.fixed_start is not None:
            start = self.fixed_start
        elif start is None:
            start = np.random.randint((len(self.data) // 24) - 1) * 24

        self.start = start

        end = start + self.num_steps + 1

        self.episode_values = self.data[start:end]

    def step(self) -> None:
        self.time_step += 1

    def get_next_load(self) -> float:
        """Get power load for next time step.

        Returns:
            float: load power (kW)
        """
        load_power = self.episode_values[self.time_step]
        self.step()
        return load_power

    def get_prediction(self, start_time: float, end_time: float) -> np.array:
        """Get prediction of future load.

        Args:
            start_time (float): begin of prediction
            end_time (float): end of prediction

        Returns:
            np.array: predicted load (kW)
        """
        return self.episode_values[start_time:end_time]

    def fix_start(self, start: int = 0) -> None:
        if start is None:
            self.fixed_start = None
        elif start > len(self.data) // 24:
            raise ValueError("Higher start index than days in data.")
        else:
            self.fixed_start = start * 24
