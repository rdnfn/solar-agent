"""This module contains electrical grid models."""


class GridModel:
    """Base class for grid models."""

    def transfer(self, power: float) -> float:
        """Transfer power to the grid.

        Returns the price paid to or from home owner for either receiving
        or giving power to the grid at time t.

        Args:
            power (float): power to transfer (kW)

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


class PeakGrid(GridModel):
    """Grid model with peak-demand pricing."""

    def __init__(
        self,
        peak_threshold: float = 1.6,
        base_price: float = 0.14,
        peak_price: float = 1.0,
        time_step_len: float = 1.0,
    ) -> None:
        """Grid model with peak-demand pricing.

        Args:
            peak_threshold (float): amount of power drawn above which the higher
                peak price is charged (in kW).
            base_price (float): price charged normally ($/kWh).
            peak_price (float): price charged if peak threshold is passed ($/kWh).
            time_step_len (float): duration of time step (in h).
        """
        self.peak_threshold = peak_threshold
        self.peak_price = peak_price
        self.base_price = base_price
        self.time_step_len = time_step_len

    def draw_power(self, power: float) -> float:
        """Transfer power to the grid.

        Returns the price paid to or from home owner for either receiving
        or giving power to the grid at time t.

        Args:
            power (float): power to transfer (kW)
            time (float): time of transfer

        Returns:
            float: price paid (can be positive or negative)
        """
        if power < 0:
            raise ValueError("Peak grid model can't accept incoming power (power<0).")

        if power > self.peak_threshold:
            return (
                self.peak_threshold * self.base_price
                + (power - self.peak_threshold) * self.peak_price
            ) * self.time_step_len
        else:
            return power * self.base_price * self.time_step_len
