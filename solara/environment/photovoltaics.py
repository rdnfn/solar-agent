"""This module contains photovoltaic system models."""


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
