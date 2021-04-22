"""This module contains photovoltaic system models."""

import numpy as np


class BasePV:
    """Base class for photovoltaic (PV) system models."""

    def __init__(self) -> None:
        """Base class for photovoltaic (PV) system models."""
        raise NotImplementedError

    def get_generation(self, time: float) -> float:
        """Get power generation for given time.

        Args:
            time (float): time of generation

        Returns:
            float: generated power (kW)
        """
        raise NotImplementedError

    def get_prediction(self, start_time: float, end_time: float) -> np.array:
        """Get prediction of future PV generation.

        Args:
            start_time (float): begin of prediction
            end_time (float): end of prediction

        Returns:
            np.array: predicted generation (kW)
        """
        raise NotImplementedError
