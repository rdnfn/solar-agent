"""Module containing constants."""
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

DEFAULT_ENV_CONFIG = {
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
