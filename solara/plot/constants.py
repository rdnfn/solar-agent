"""Constants configuring plots."""

LABELS = {
    "load": "Load (kW)",
    "pv_gen": "Solar gen. (kW)",
    "battery_cont": "Energy cont. (kWh)",
    "actions": "Actions",
    "rewards": "Rewards ($)",
    "net_load": "Net load (kW)",
    "cost": "Cost ($)",
    "price_threshold": "Price threshold ($)",
    "charging_power": "Charging power (kW)",
    "power_diff": "Infeas. control (kW)",
    "time_step": "Time step (h)",
    "cum_load": "Cum. load (kW)",
    "cum_pv_gen": "Cum. PV gen. (KW)",
}

COLORS = {
    "load": "blue",
    "pv_gen": "green",
    "energy_cont": "black",
    "actions": "red",
    "rewards": "purple",
    "charging_power": "orange",
    "price_threshold": (0.3, 0.3, 0.3, 0.3),
}

MARKERS = {
    "price_threshold": None,
}
