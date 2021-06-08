"""Module with standard environment configs."""

from solara.constants import PROJECT_PATH

EXP001_SINGLE_DATA = {
    "general": {
        "type": "battery_control.BatteryControlEnv",
        "infeasible_control_penalty": True,
        "grid_charging": True,
        "logging_level": "WARNING",  # if using RLlib, set to 'RAY'
    },
    "components": {
        "battery": {
            "type": "LithiumIonBattery",
            "size": 10,
            "chemistry": "NMC",
            "time_step_len": 1,
        },
        "solar": {
            "type": "DataPV",
            "data_path": PROJECT_PATH + "/data/solar_trace_data/PV_5796.txt",
            "fixed_sample_num": 12,
        },
        "load": {
            "type": "DataLoad",
            "data_path": PROJECT_PATH + "/data/solar_trace_data/load_5796.txt",
            "fixed_sample_num": 12,
        },
        "grid": {
            "type": "PeakGrid",
            "peak_threshold": 1.0,
        },
    },
}


DEFAULT_ENV_CONFIG = EXP001_SINGLE_DATA
