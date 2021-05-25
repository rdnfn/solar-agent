"""This module contains functions that visualise solar agent control."""
from __future__ import annotations
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from solara.plot.constants import COLORS, LABELS, MARKERS


def plot_episode(
    data: Dict[str, np.array],
    colors: Dict[str, str] = None,
    labels: Dict[str, str] = None,
    markers: Dict[str, str] = None,
    selected_keys: List[str] = None,
    num_timesteps: int = 25,
    iteration: int = None,
    title: str = "Episode Trajectory",
    y_max: float = 4,
    y_min: float = -2.5,
    show_grid: bool = True,
    figsize: Tuple = (6, 6),
    rewards_key: str = "rewards",
    dpi: int = 100,
):
    """Plot a single episode of battery control problem."""

    if colors is None:
        colors = COLORS

    if labels is None:
        labels = LABELS

    if markers is None:
        markers = MARKERS

    x = np.arange(0, num_timesteps)

    if rewards_key in data.keys():
        episode_reward = sum(data[rewards_key])
    else:
        episode_reward = None

    # Setting up the figure
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xticks([0, 5, 10, 15, 20, 23], minor=False)
    ax.set_xticks(x, minor=True)
    ax.set_xticklabels([0, 5, 10, 15, 20, 23], minor=False)

    if show_grid:
        ax.yaxis.grid(True, which="major")
        ax.xaxis.grid(True, which="major")
        ax.xaxis.grid(True, which="minor")
    # ax.set_prop_cycle("color", colors)

    # Plotting the data
    for name, values in data.items():
        if selected_keys is None or name in selected_keys:
            if name in colors.keys():
                color = colors[name]
            else:
                color = None

            if name in labels.keys():
                label = labels[name]
            else:
                label = name

            if name in markers.keys():
                marker = markers[name]
            else:
                marker = "."

            ax.plot(values, label=label, marker=marker, color=color)

    if title is not None:
        if iteration is not None:
            iteration_str = "Iteration {:2.0f}, ".format(iteration)
        else:
            iteration_str = ""
        if episode_reward is not None:
            title += "    ({}Overall reward: {:.3f})".format(
                iteration_str, episode_reward
            )

        plt.title(title)
    plt.ylabel("kW / kWh / other")
    plt.legend()

    plt.ylim(ymin=y_min, ymax=y_max)

    plt.show()
