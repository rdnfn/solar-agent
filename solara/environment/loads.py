"""This module contains residential load models."""


class BaseLoadModel:
    """Base class for residential load models."""

    def __init__(self) -> None:
        """Base class for residential load models."""
        raise NotImplementedError

    def get_load(self, time: float) -> float:
        """Get power load for given time.

        Args:
            time (float): time of generation

        Returns:
            float: power consumed (kW)
        """
        raise NotImplementedError
