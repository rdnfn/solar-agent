"""Module with different types of batteries."""

from typing import List, Tuple

import numpy as np
import cvxpy as cp

from solara.envs.components.base import EnvComponent


class BatteryModel(EnvComponent):
    """Base battery model."""


class LithiumIonBattery(BatteryModel):
    """Class modelling lithium-ion battery."""

    def __init__(self, size: float, chemistry: str, time_step_len: float):
        """Class modelling lithium-ion battery.

        This class was originally written and kindly provided by Fiodar Kazhamiaka.
        It has been adapted from the original version. The class implements the
        C/L/C model described on page 12 of
        https://cs.stanford.edu/~fiodar/pubs/TractableLithium-ionStorageMod.pdf.

        Args:
            size (float): kWh capacity.
            chemistry (str): chemistry type of the battery. One of
                "LTO" (lithium titanate) or "NMC" (Li Nickel Manganice Cobalt).
            time_step_len (float): time step length/duration of simulation in hours
                (originally named T_u).
        """
        super().__init__()

        self.size = size
        self.chemistry = chemistry
        self.time_step_len = time_step_len

        # declare battery model parameters
        self.num_cells = None
        self.kWh_per_cell = None  # pylint: disable=invalid-name
        self.nominal_voltage_c = None
        self.nominal_voltage_d = None

        self.u1 = None
        self.v1_bar = None

        self.u2 = None
        self.v2_bar = None

        self.eta_d = None
        self.eta_c = None

        self.alpha_bar_d = None
        self.alpha_bar_c = None

        self.set_parameters()

        # battery energy content
        self.b = self.v1_bar

    def set_parameters(self):
        """Set battery model parameters according to specified Li-Ion chemistry."""

        if self.chemistry == "NMC":

            self.kWh_per_cell = 0.011284
            self.num_cells = self.size / self.kWh_per_cell

            # parameters specified for an LNMC cell with operating range of 1 C
            # charging and discharging
            self.nominal_voltage_c = 3.8793
            self.nominal_voltage_d = 3.5967
            self.u1 = 0.1920
            self.v1_bar = 0.0
            self.u2 = -0.4865
            self.v2_bar = self.kWh_per_cell * self.num_cells
            self.eta_d = 1 / 0.9  # taking reciprocal so that we don't divide by eta_d
            self.eta_c = 0.9942
            self.alpha_bar_d = (
                self.v2_bar * 1
            )  # the 1 indicates the maximum discharging C-rate
            self.alpha_bar_c = (
                self.v2_bar * 1
            )  # the 1 indicates the maximum charging C-rate

        elif self.chemistry == "LTO":

            self.kWh_per_cell = 0.0739108
            self.num_cells = self.size / self.kWh_per_cell
            self.nominal_voltage_c = 2.3624
            self.nominal_voltage_d = 2.0759
            self.u1 = 0.1559
            self.v1_bar = 0.0
            self.u2 = -0.0351
            self.v2_bar = self.kWh_per_cell * self.num_cells
            self.eta_d = (
                1 / 0.9716
            )  # taking reciprocal so that we don't divide by eta_d
            self.eta_c = 0.9741
            self.alpha_bar_d = self.v2_bar * 2
            self.alpha_bar_c = self.v2_bar * 2

        else:
            print("chemistry is not supported")

    def calc_max_charging(self, power: float) -> float:
        """Calculate the maximum amount of charging possible.

        Decrease the applied (charging) power by increments of (1/30) until the power is
        low enough to avoid violating the upper energy limit constraint.
        Could speed these functions up with a binary search instead of linear search,
        or a lookup table.

        Args:
            power (float): applied charging power (in kW)

        Returns:
            float: max amount of power that can be charged.
        """
        # Implements constraint (4)
        for c in np.linspace(power, 0, num=30):
            upper_lim = self.u2 * (c / self.nominal_voltage_c) + self.v2_bar
            b_temp = self.b + c * self.eta_c * self.time_step_len
            if b_temp <= upper_lim:
                return c
        return 0

    def calc_max_discharging(self, power: float) -> float:
        """Calculate the maximum amount of discharging possible.

        Decrease the applied (discharging) power by increments of (1/30) until the power
        is low enough to avoid violating the lower energy limit constraint.

        Args:
            power (float): power to be discharged (kW)

        Returns:
            float: max power that can be discharged.
        """
        # Implements constraint (4)
        for d in np.linspace(power, 0, num=30):
            lower_lim = self.u1 * (d / self.nominal_voltage_d) + self.v1_bar
            b_temp = self.b - d * self.eta_d * self.time_step_len
            if b_temp >= lower_lim:
                return d
        return 0

    def get_charging_limits(self) -> Tuple[float, float]:
        """Get general maximum and minimum charging constraints."""
        max_charge_power = self.size
        min_charge_power = -self.size

        return min_charge_power, max_charge_power

    def charge(self, power: float) -> float:
        """Update the battery's energy content.

        This method updates the battery's energy content (b) according to
        charging/discharging power (p).

        Args:
            power (float): power to charge or discharge (kW)
        """
        new_c = 0
        new_d = 0

        # clip power so that it satisfies charging rate constraints
        if power > 0:
            new_c = min(power, self.alpha_bar_c)  # Implements constraint (3)
            new_c = self.calc_max_charging(new_c)
            new_d = 0
        elif power < 0:
            new_d = min(-power, self.alpha_bar_d)  # Implements constraint (3)
            new_d = self.calc_max_discharging(new_d)
            new_c = 0

        # Implements contraint (1) and (2)
        self.b = (
            self.b
            + new_c * self.eta_c * self.time_step_len
            - new_d * self.eta_d * self.time_step_len
        )

        # actual amount of power applied
        actual_power = new_c - new_d

        self.logger.debug(
            "Charged %6.4fkW (attempted %6.4f), new content %2.2fkWh",
            actual_power,
            power,
            self.b,
        )

        return actual_power

    def get_energy_content(self) -> None:
        """Return the current energy content."""
        return self.b

    def reset(self) -> None:
        """Reset battery energy content."""
        self.b = self.v1_bar

    def get_contraints(
        self, power_episode: cp.Variable, battery_content_episode: cp.Variable
    ) -> List:
        """Return CVXPY-compatible list of constraints.

        This follows the C/L/C model described on page 12 of
        https://cs.stanford.edu/~fiodar/pubs/TractableLithium-ionStorageMod.pdf
        """

        # constraint in Equation (20) of paper above
        # TODO move into contraints
        delta_energy = cp.multiply(
            power_episode * self.eta_c * self.time_step_len, (power_episode >= 0)
        )
        delta_energy += cp.multiply(
            power_episode * self.eta_d * self.time_step_len, (power_episode < 0)
        )

        constraints = [
            battery_content_episode[1:]
            == battery_content_episode[:-1]
            + delta_energy,  # constraint in Equation (19)
            # constraint in Equation (5)
            self.alpha_bar_d <= power_episode,
            self.alpha_bar_c >= power_episode,
            # constraint in Equation (22)
            self.u1 * power_episode / self.nominal_voltage_d + self.v1_bar
            <= battery_content_episode,
            self.u2 * power_episode / self.nominal_voltage_c + self.v2_bar
            >= battery_content_episode,
        ]
        return constraints
