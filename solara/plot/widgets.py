"""Widgets for visualising episodes in environments."""

from typing import Dict, List

import ipywidgets as widgets

import solara.plot.pyplot


class InteractiveEpisodes(widgets.HBox):
    """Interactive episode widget."""

    def __init__(self, episode_data: List[Dict]) -> None:
        """Interactive episode widget.

        Args:
            episode_data (List[Dict]): list of dictionaries containing episode data.
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

        checkboxes = self._create_visibility_checkboxes()

        checkbox_box = widgets.Accordion(
            children=[widgets.VBox([*checkboxes.values()])],
        )
        checkbox_box.set_title(index=0, title="Visibility")

        out = widgets.interactive_output(
            self._plot_episode,
            dict(
                iteration=slider, episode_data=widgets.fixed(episode_data), **checkboxes
            ),
        )
        children = [widgets.VBox([widgets.HBox([play, slider]), out]), checkbox_box]

        super().__init__(children=children)

    def _create_visibility_checkboxes(self) -> object:
        checkboxes = {}
        for key in self.episode_data[0].keys():
            checkbox = widgets.Checkbox(
                value=True, description=key, disabled=False, indent=False
            )
            checkboxes[key] = checkbox

        return checkboxes

    def _plot_episode(
        self, episode_data: List[Dict] = None, iteration: int = 10, **kwargs
    ) -> None:
        """[summary]

        [extended_summary]

        Args:
            episode_data (List[Dict], optional): [description]. Defaults to None.
            iteration (int, optional): [description]. Defaults to 10.
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
        )

        self._print_episode_data(single_episode_data)

    def _print_episode_data(self, single_episode_data: Dict) -> None:
        print(
            "Overall - Rewards: {:.3f}, Cost: {:.3f}, Power_diff: {:.3f}".format(
                sum(single_episode_data["rewards"]),
                sum(single_episode_data["cost"]),
                sum(single_episode_data["power_diff"]),
            )
        )
