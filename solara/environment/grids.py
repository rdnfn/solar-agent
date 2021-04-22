"""This module contains electrical grid models."""


class BaseGridModel:
    """Base class for grid models."""

    def __init__(self) -> None:
        """Base class for grid models."""
        raise NotImplementedError

    def transfer(self, power: float, time: float) -> float:
        """Transfer power to the grid.

        Returns the price paid to or from home owner for either receiving
        or giving power to the grid at time t.

        Args:
            power (float): power to transfer (kW)
            time (float): time of transfer

        Returns:
            float: price paid (can be positive or negative)
        """
        raise NotImplementedError

    def get_sell_price(self, time: float) -> float:
        """Return the price paid for selling in the grid.

        Args:
            time (float): time of price

        Returns:
            float: price paid
        """
        raise NotImplementedError

    def get_buy_price(self, time: float) -> float:
        """Return the price paid when buying in the grid.

        Args:
            time (float): time of price

        Returns:
            float: price paid
        """
        raise NotImplementedError
