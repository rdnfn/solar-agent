"""This module contains functions that visualise solar agent control."""
from __future__ import annotations
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from solara.plot.constants import COLORS, LABELS, MARKERS


def default_setup(figsize=None) -> None:
    """Setup default matplotlib settings."""

    if figsize is None:
        figsize = (6, 3)

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)
    sns.set_style("ticks", {"dashes": False})
    sns.set_context("paper")


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
    figsize: Tuple = (4.62, 3),
    rewards_key: str = "rewards",
    dpi: int = 100,
    include_episode_stats: bool = True,
):
    """Plot a single episode of battery control problem."""

    # default_setup()

    matplotlib.rc("text", usetex=True)

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

            label = label.replace("$", "\\$")

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
    plt.xlabel("Time step")

    # Adding overall data
    if "power_diff" in data:
        power_diff_sum = float(sum(data["power_diff"]))
    else:
        power_diff_sum = 0

    handles, _ = ax.get_legend_handles_labels()

    if include_episode_stats:
        ep_summary_stats = (
            # "\\rule{{67pt}}{{0.25pt}}"
            "\n \\textbf{{Episode statistics}}"
            "\n Sum of rewards: {:>8.3f} \\\\"
            "\n Sum of costs:  {:>15.3f} \\\\"
            "\n Sum of penalties: {:>11.3f}"
        ).format(
            float(sum(data["rewards"])),
            float(sum(data["cost"])),
            power_diff_sum,
        )

        handles.append(matplotlib.patches.Patch(color="none", label=ep_summary_stats))

    plt.legend(
        bbox_to_anchor=(1.02, 1.025),
        loc="upper left",
        edgecolor="grey",
        handles=handles,
        # title="\\textbf{{Legend}}",
    )

    plt.ylim(ymin=y_min, ymax=y_max)

    # plt.show()
