"""Widgets for visualising episodes in environments."""

from typing import Dict, List, Tuple

import ipywidgets as widgets

import solara.plot.pyplot
from solara.plot.constants import LABELS


class InteractiveEpisodes(widgets.HBox):
    """Interactive episode widget."""

    def __init__(
        self, episode_data: List[Dict], initial_visibility: List[str] = None
    ) -> None:
        """Interactive episode widget.

        Args:
            episode_data (List[Dict]): list of dictionaries containing episode data.
            initial_visibility (List[str], optional): which values to plot initially.
                Defaults to None, which makes all lines visible.
        """

        self.episode_data = episode_data

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

        checkboxes = self._create_visibility_checkboxes(initial_visibility)
        y_range_slider = widgets.FloatRangeSlider(
            value=[-2.5, 4],
            min=-10,
            max=10.0,
            step=0.1,
            description="y-axis range",
            disabled=False,
            continuous_update=False,
            orientation="vertical",
            readout=True,
            readout_format=".1f",
        )

        checkbox_box = widgets.Accordion(
            children=[widgets.VBox([*checkboxes.values()]), y_range_slider],
        )
        checkbox_box.set_title(index=0, title="Visibility")
        checkbox_box.set_title(index=1, title="Other settings")

        out = widgets.interactive_output(
            self._plot_episode,
            dict(
                iteration=slider,
                episode_data=widgets.fixed(episode_data),
                y_range=y_range_slider,
                **checkboxes,
            ),
        )
        children = [widgets.VBox([widgets.HBox([play, slider]), out]), checkbox_box]

        super().__init__(children=children)

    def _create_visibility_checkboxes(self, initial_visibility: List[str]) -> Dict:
        """Create visibility checkboxes.

        This method creates a checkbox for each (potential) plotted line in the widget.

        Returns:
            Dict: dictionary of checkbox widgets
        """
        checkboxes = {}
        for key in self.episode_data[0].keys():
            if key in LABELS.keys():
                label = LABELS[key]
            else:
                label = key

            if initial_visibility is None:
                value = True
            else:
                value = bool(key in initial_visibility)

            checkbox = widgets.Checkbox(
                value=value, description=label, disabled=False, indent=False
            )
            checkboxes[key] = checkbox

        return checkboxes

    def _plot_episode(
        self,
        episode_data: List[Dict] = None,
        iteration: int = 10,
        y_range: Tuple[float, float] = (-2.5, 4),
        **kwargs
    ) -> None:
        """Plot an episode for a given iteration.

        Args:
            episode_data (List[Dict], optional): data from multiple episodes.
                Defaults to None.
            iteration (int, optional): iteration to plot. Defaults to 10.
            y_range (Tuple[float, float], optional): Range of y axis in plot.
                Defaults to (-2.5, 4).
        """

        selected_keys = []
        for key, selected in kwargs.items():
            if selected:
                selected_keys.append(key)

        single_episode_data = episode_data[iteration - 1]

        solara.plot.pyplot.plot_episode(
            single_episode_data,
            show_grid=False,
            selected_keys=selected_keys,
            title=None,
            y_min=y_range[0],
            y_max=y_range[1],
        )

        self._print_episode_data(single_episode_data)

    def _print_episode_data(self, single_episode_data: Dict) -> None:
        print(
            "Overall - Rewards: {:.3f}, Cost: {:.3f}, Power_diff: {:.6f}".format(
                sum(single_episode_data["rewards"]),
                sum(single_episode_data["cost"]),
                sum(single_episode_data["power_diff"]),
            )
        )
