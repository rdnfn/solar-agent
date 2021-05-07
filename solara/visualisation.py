"""This module contains functions that visualise solar agent control."""

import numpy as np
import matplotlib.pyplot as plt


def plot_battery_control(
    energy_content_trace: np.array,
    pv_trace: np.array,
    load_trace: np.array,
    save_path: str = None,
) -> None:
    """Plot battery control.

    This function creates a simple plot comparing battery energy content,
    PV generation and load over time.

    Args:
        energy_content_trace (np.array): trace of energy content (kWh)
        pv_trace (np.array): trace of PV generation (kW)
        load_trace (np.array): trace of residential load (kW)
        save_path (str, optional): optional path to save plot. Defaults to None.
    """

    time_array = np.arange(len(energy_content_trace))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), dpi=150)
    fig.suptitle("PV Battery Control")

    ax1.plot(time_array, pv_trace, ".-")
    ax1.set_ylabel("PV trace (kW)")

    ax2.plot(time_array, load_trace, ".-")
    ax2.set_ylabel("Load (kW)")

    ax3.plot(time_array, energy_content_trace, ".-")
    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Battery content (kWh)")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
