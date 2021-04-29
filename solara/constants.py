"""Module containing constants."""
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

# Notation constants
# Each notation is a tuple of (math_notation, py_notation, description, unit)

# Notation similar to
# https://cs.stanford.edu/~fiodar/pubs/TractableLithium-ionStorageMod.pdf
# Notable exceptions: using kW(h) instead of W(h)
battery_notation = [
    ("P(k)", "power", "Power applied to battery in the current time step ($k$)", "kW"),
    (
        "\\{P(k)\\}_{0 \\leq k \\leq T}",
        "power_episode",
        "Power applied to battery throughout an episode",
        "kW",
    ),
    ("I(k)", "current", "Current estimate in the current time step ($k$)", "A"),
    (
        "V(k)",
        "voltage",
        "Battery voltage estimate in the current time step ($k$) ",
        "V",
    ),
    (
        "b(k)",
        "battery_content",
        "Battery energy content estimate at the end of the current time step ($k$)",
        "kWh",
    ),
    (
        "\\{b(k)\\}_{0 \\leq k \\leq T}",
        "battery_content_episode",
        "Battery energy content estimate throughout an episode",
        "kWh",
    ),
    ("a_1", "min_cell_content", "Minimum cell energy content", "kWh"),
    ("a_2", "max_cell_content", "Maximum cell energy content", "kWh"),
    (
        "alpha_c",
        "charge_current_limit",
        "Charge current limits per unit of storage",
        "A/kWh",
    ),
    (
        "alpha_d",
        "discharge_current_limit",
        "Discharge current limits per unit of storage",
        "A/kWh",
    ),
    ("eta_c", "charge_efficiency", "Charge efficiency (0 <= eta_c <=1)", ""),
    ("eta_d", "discharge_efficiency", "Discharge efficiency (0 <= eta_d <=1)", ""),
    ("n", "num_cells", "Number of cells in battery"),
    (
        "R_{ic }",
        "charge_cell_impedance",
        "Internal cell impedance during charging",
        "\\Omega",
    ),
    (
        "R_{id }",
        "discharge_cell_impedance",
        "Internal cell impedance during charging",
        "\\Omega",
    ),
    ("T_u", "time_step_len", "Length of a single time step/slot", "hours"),
    ("V_{nom,c}", "charge_nominal_voltage", "Nominal voltage when charging", "V"),
    ("V_{nom,d}", "discharge_nominal_voltage", "Nominal voltage when discharging", "V"),
    ("", "cell_size", "Energy content capacity per battery cell", "kWh"),
    ("", "size", "Energy content capacity of entire battery", "kWh"),
]

# Notation from
# https://cs.stanford.edu/~fiodar/papers/practical-strategies-storage-20.pdf
battery_notation += [
    ("B", "batt_capacity", "Capacity of battery", "kWh"),
    ("MD", "batt_max_discharge", "Maximum discharge fraction of battery", ""),
    ("MC", "batt_max_charge", "Maximum charge fraction of battery", ""),
]
