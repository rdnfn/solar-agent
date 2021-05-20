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
            solara.plot.pyplot.plot_episode(episode_data[iteration], show_grid=False)

        play = widgets.Play(
            value=0,
            min=0,
            max=10,
            step=1,
            interval=400,
            description="Press play",
            disabled=False,
        )
        slider = widgets.IntSlider(0, 0, 10)
        widgets.jslink((play, "value"), (slider, "value"))

        out = widgets.interactive_output(
            plot_episode,
            dict(iteration=slider, episode_data=widgets.fixed(episode_data)),
        )
        children = [widgets.HBox([play, slider]), out]

        super().__init__(children=children)
