"""Module defining project notation."""

from __future__ import annotations

from dataclasses import dataclass, InitVar
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import solara.envs.wiring


@dataclass
class VarDef:
    """Definition of notation for a single variable."""

    var_name: str
    latex_math: str = " "
    unit: str = ""
    description: str = ""
    latex_cmd: InitVar[str] = None

    def __post_init__(self, latex_cmd):
        """Complete init."""
        if latex_cmd is not None:
            self.latex_cmd = latex_cmd
        else:
            self.latex_cmd = "\\" + self.var_name.replace("_", "")


@dataclass
class NotationCollection:
    """Notation collection.

    Class to collect and display notation.
    """

    notation_list: List[VarDef]

    def print_notation_style(self) -> None:
        """Print notation as latex style file commands."""
        for variable in self.notation_list:
            print(
                "\\newcommand{{{}}}{{{}}}".format(
                    variable.latex_cmd, variable.latex_math
                )
            )

    def get_latex_table_str(self) -> str:
        """Get notation table formatted in latex.

        Returns:
            str: latex string
        """

        out = ""
        out += r"\begin{center}" + "\n"
        out += r"\begin{tabular}{ l p{6cm} l l}" + "\n"
        out += "Variable & Description & Unit & Python Name \\\\ \n"
        out += "\\hline"

        for variable in self.notation_list:
            out += "${}$ & {} & {} & \\texttt{{{}}} \\\\".format(
                variable.latex_math,
                variable.description,
                variable.unit,
                variable.var_name.replace("_", r"\_"),
            )
            out += "\n"

        out += r"\end{tabular}" + "\n"
        out += r"\end{center}"

        return out

    def get_mrkdwn_table_str(self) -> str:
        """Get notation table formatted in markdown.

        Returns:
            str: markdown string
        """

        out = ""
        out += "Variable | Description | Unit | Python Name \n"
        out += "---|---|---|--- \n"

        for variable in self.notation_list:
            out += "${}$ | {} | {} | `{}`".format(
                variable.latex_math,
                variable.description,
                variable.unit,
                variable.var_name,
            )
            out += "\n"

        return out


def create_power_variables(power_flow: solara.envs.wiring.PowerFlow) -> list:
    """Create a list of notation variable definitions from an electric system."""
    var_defs = []
    for component in power_flow.components:

        # overall power
        var_name = "power_flow['{}']".format(component)
        latex_math = "P_{}".format(power_flow.component_abbr[component])
        unit: str = "kW"
        description: str = "power input(negative)/ output(positive) of {}".format(
            component
        )
        latex_cmd = "\\powerflow{}".format(component)

        var_defs.append(
            VarDef(var_name, latex_math, unit, description, latex_cmd=latex_cmd)
        )

        # input
        var_name = "-min(power_flow['{}'], 0)".format(component)
        latex_math = "P_{{\\rightarrow {}}}".format(
            power_flow.component_abbr[component]
        )
        unit: str = "kW"
        description: str = "power input to {}".format(component)
        latex_cmd = "\\powerin{}".format(component)

        var_defs.append(
            VarDef(var_name, latex_math, unit, description, latex_cmd=latex_cmd)
        )

        # output
        var_name = "max(power_flow['{}'], 0)".format(component)
        latex_math = "P_{{{}\\rightarrow }}".format(
            power_flow.component_abbr[component]
        )
        unit: str = "kW"
        description: str = "power output from {}".format(component)
        latex_cmd = "\\powerout{}".format(component)

        var_defs.append(
            VarDef(var_name, latex_math, unit, description, latex_cmd=latex_cmd)
        )

    for connection in power_flow.get_connections():
        source_cmp, target_cmp = connection
        var_name = "power_flow['{}','{}']".format(source_cmp, target_cmp)
        latex_math = "P_{{{}{}}}".format(
            power_flow.component_abbr[source_cmp], power_flow.component_abbr[target_cmp]
        )
        unit: str = "kW"
        description: str = "power transferred from {} to {}".format(
            source_cmp, target_cmp
        )
        latex_cmd = "\\power{}to{}".format(source_cmp, target_cmp)

        var_defs.append(
            VarDef(var_name, latex_math, unit, description, latex_cmd=latex_cmd)
        )

    return var_defs


_NOTATION_LIST = [
    # Power variables
    VarDef("power_charge", r"P_\text{c}", "kW", "power used to charge the battery"),
    VarDef("power_discharge", r"P_\text{d}", "kW", "power discharged from the battery"),
    VarDef("power_solar", r"P_{\text{solar}}", "kW", "power coming from solar panels"),
    VarDef("power_load", r"P_{\text{load}}", "kW", "power used by residential load"),
    VarDef("power_sell", r"P_\text{sell}", "kW", "power sold to the grid"),
    VarDef("power_grid", r"P_\text{grid}", "kW", "power drawn from the grid"),
    VarDef(
        "power_direct",
        r"P_\text{direct}",
        "kW",
        "sum of power from solar panels and grid that is used for load or sold",
    ),
    VarDef(
        "power_over_thresh", r"P_\text{over}", "kW", "power over peak demand threshold"
    ),
    # Battery
    VarDef("energy_battery", r"E_\text{batt}", "kWh", "energy content of the battery"),
    VarDef("size", r"B", "kWh", "energy capacity of battery"),
    VarDef(
        "kWh_per_cell", r"B_\text{cell}", "kWh", "energy capacity per individual cell"
    ),
    VarDef("num_cells", r"n_\text{cell}", "cells", "number of cells in battery"),
    VarDef(
        "nominal_voltage_c",
        r"V_{\text{nom},c}",
        "V",
        "nominal voltage of battery when charging",
    ),
    VarDef(
        "nominal_voltage_d",
        r"V_{\text{nom},d}",
        "V",
        "nominal voltage of battery when discharging",
    ),
    # Grid
    VarDef(
        "price_base",
        r"\pi_b",
        r"\$/kWh",
        "base price paid for energy drawn from the grid",
    ),
    VarDef(
        "price_penalty",
        r"\pi_d",
        r"\$/kWh",
        (
            "additional price penalty paid for energy drawn"
            " from the grid when demand is above threshold"
        ),
    ),
    VarDef(
        "grid_threshold",
        r"\Gamma",
        "kW",
        "demand threshold above which price penalty is paid",
    ),
    VarDef(
        "eff_discharge",
        r"\eta_d",
        "kWh",
        (
            "efficiency of discharging the battery, amount of energy"
            " content reduction for discharging 1 kWh"
        ),
    ),
    VarDef(
        "eff_charge",
        r"\eta_c",
        "kWh",
        (
            "efficiency of charging the battery, amount of"
            " energy content increase for charging 1 kWh"
        ),
    ),
    # General parameters
    VarDef("num_timesteps", r"T", "steps", "number of time steps in an episode"),
    VarDef("len_timestep", r"\delta_\text{step}", "hours", "length of a timestep"),
]

NOTATION = NotationCollection(_NOTATION_LIST)
