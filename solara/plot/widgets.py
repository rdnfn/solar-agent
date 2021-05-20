"""Widgets for visualising episodes in environments."""

from typing import Dict, List

import ipywidgets as widgets

import solara.plot.pyplot


class InteractiveEpisodes(widgets.VBox):
    """Interactive episode widget."""

    def __init__(self, episode_data: List[Dict]) -> None:
        """Interactive episode widget.

        Args:
            episode_data (List[Dict]): list of dictionaries containing episode data.
        """

        def plot_episode(episode_data: List[Dict] = None, iteration: int = 10) -> None:
            solara.plot.pyplot.plot_episode(
                episode_data[iteration - 1], show_grid=False
            )

        num_episodes = len(episode_data)

        play = widgets.Play(
            value=1,
            min=1,
            max=num_episodes,
            step=1,
            interval=400,
            disabled=False,
        )
        slider = widgets.IntSlider(1, 1, num_episodes)
        widgets.jslink((play, "value"), (slider, "value"))

        out = widgets.interactive_output(
            plot_episode,
            dict(iteration=slider, episode_data=widgets.fixed(episode_data)),
        )
        children = [widgets.HBox([play, slider]), out]

        super().__init__(children=children)
