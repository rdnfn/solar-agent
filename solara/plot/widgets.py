"""Widgets for visualising episodes in environments."""

from typing import Dict, List, Tuple

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

import solara.plot.pyplot
import solara.utils.rllib
from solara.plot.constants import LABELS


class InteractiveEpisodes(widgets.HBox):
    """Interactive episode widget."""

    def __init__(
        self,
        episode_data: List[Dict] = None,
        initial_visibility: List[str] = None,
        manual_mode: bool = False,
        manual_start_actions: List = None,
        env: object = None,
        plot_figsize: Tuple = (6, 4),
    ) -> None:
        """Interactive episode widget.

        Args:
            episode_data (List[Dict]): list of dictionaries containing episode data.
                Defaults to None.
            initial_visibility (List[str], optional): which values to plot initially.
                Defaults to None, which makes all lines visible.
            manual_mode (bool, optional): Whether to manually adapt policy.
                Defaults to False.
            manual_start_actions (List, optional): Starting values for manual policy.
                Only relevant if `manual_mode == True`. Defaults to None.
            env (object, optional): environment to be used in manual mode.
                Defaults to None.
            plot_figsize (Tuple, optional): size of plot shown, arg is passed to
                matplotlib. Defaults to (6, 4).
        """

        # Create output widgets (log and plot)
        self.widgets = {
            "plot": widgets.Output(),
            "log_out": widgets.Output(layout={"border": "1px solid black"}),
        }

        self.episode_data = episode_data
        self.manual_mode = manual_mode
        self.manual_start_actions = manual_start_actions
        self.plot_figsize = plot_figsize

        # Setup either manual action sliders or iteration selection play bar
        if not manual_mode:
            self.episode_data = episode_data
            num_iterations = len(episode_data)
            top_widget = self._create_play_bar(num_iterations=num_iterations)

            # Hiding top bar if only a single episode in data
            if len(self.episode_data) == 1:
                hidden_layout = widgets.Layout(display="none")
                for child in top_widget.children:
                    child.layout = hidden_layout
        else:
            self.env = env
            sliders = self._create_manual_sliders()
            top_widget = widgets.HBox([*sliders.values()])

        settings = self._create_settings_accordion(initial_visibility)

        # Limit the width of top widget (mainly affects manual action sliders)
        top_widget.layout = widgets.Layout(max_width="{}in".format(plot_figsize[0]))

        self._enable_observing()
        self._plot_episode({})

        # Set overall layout of widgets
        children = [widgets.VBox([top_widget, self.widgets["plot"]]), settings]

        super().__init__(children=children)

    def _create_play_bar(self, num_iterations: int) -> widgets.Widget:
        """Create play bar widget over iterations.

        Args:
            num_iterations (int): length of the play bar

        Returns:
            widgets.Widget: play bar widget
        """
        play = widgets.Play(
            value=1,
            min=1,
            max=num_iterations,
            step=1,
            interval=400,
            disabled=False,
        )
        slider = widgets.IntSlider(1, 1, num_iterations)

        # Linking slider and play bar values in javascript
        widgets.jslink((play, "value"), (slider, "value"))

        self.widgets["iteration"] = slider

        return widgets.HBox([play, slider])

    def _create_settings_accordion(
        self, initial_visibility: List[str]
    ) -> widgets.Widget:
        """Create settings accordion widget.

        Args:
            initial_visibility (List[str]): keys of data to be plotted initially

        Returns:
            widgets.Widget: settings widget
        """

        checkboxes = self._create_visibility_checkboxes(initial_visibility)

        # Create slide for range of y-axis
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
        self.widgets["y_range_slider"] = y_range_slider

        # Combine setting widgets together into accordion
        checkbox_box = widgets.Accordion(
            children=[widgets.VBox([*checkboxes.values()]), y_range_slider],
        )
        checkbox_box.set_title(index=0, title="Visibility")
        checkbox_box.set_title(index=1, title="Other settings")

        return checkbox_box

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

        self.widgets["visibility_checkboxes"] = checkboxes

        return checkboxes

    def _create_manual_sliders(self, num: int = 24) -> Dict[str, widgets.Widget]:
        """Create sliders to manually define actions.

        Args:
            num (int, optional): number of time steps to create actions for.
                Defaults to 24.

        Returns:
            Dict[str, widgets.Widget]: dictionary with one slider per time step as
                values.
        """
        sliders = {}
        for i in range(num):
            if self.manual_start_actions is not None:
                value = self.manual_start_actions[i]
            else:
                value = 0
            sliders["input_{}".format(i)] = widgets.FloatSlider(
                value=value,
                min=-1,
                max=1,
                step=0.001,
                description=str(i),
                disabled=False,
                continuous_update=False,
                orientation="vertical",
                readout=True,
                readout_format=".3f",
            )
        self.widgets["manual_sliders"] = sliders
        return sliders

    def _plot_episode(self, change: Dict) -> None:
        """Plot an episode for a given iteration.

        Args:
            change (Dict): change dictionary from ipywidgets/traitlets. Not used for
                anything but logging.
        """

        # Get all current variable values used for plotting
        y_range = self.widgets["y_range_slider"].value
        visibility_checkboxes = self.widgets["visibility_checkboxes"]
        selected_keys = []
        for key, box in visibility_checkboxes.items():
            if box.value:
                selected_keys.append(key)

        if not self.manual_mode:
            iteration = self.widgets["iteration"].value
            single_episode_data = self.episode_data[iteration - 1]
        else:
            actions = []
            for slider in self.widgets["manual_sliders"].values():
                actions.append(np.array([slider.value]))
            agent = solara.utils.rllib.DeterministicAgent(actions, self.env)
            observations, actions, rewards, infos = solara.utils.rllib.run_episode(
                agent
            )
            single_episode_data = solara.utils.rllib.get_episode_dict(
                observations,
                actions,
                rewards,
                infos,
            )

        # Re-draw the plot in "plot" output widget
        self.widgets["plot"].clear_output(wait=True)
        with self.widgets["plot"]:
            solara.plot.pyplot.plot_episode(
                single_episode_data,
                show_grid=False,
                selected_keys=selected_keys,
                title=None,
                y_min=y_range[0],
                y_max=y_range[1],
                figsize=self.plot_figsize,
            )
            plt.show()
            self._print_episode_data(single_episode_data)

        with self.widgets["log_out"]:
            print("plotted")
            print(change)

    def _print_episode_data(self, single_episode_data: Dict) -> None:
        """Print summary data of episode.

        Args:
            single_episode_data (Dict): data for single episode
        """
        if "power_diff" in single_episode_data:
            power_diff_sum = float(sum(single_episode_data["power_diff"]))
        else:
            power_diff_sum = 0

        print(
            "Overall - Rewards: {:.3f}, Cost: {:.3f}, Power_diff: {:.6f}".format(
                float(sum(single_episode_data["rewards"])),
                float(sum(single_episode_data["cost"])),
                power_diff_sum,
            )
        )

    def _enable_observing(self) -> None:
        """Enable observing of value changes in widgets and set plot callback function.

        This method enables a re-drawing of the plot whenever the value of any widget
        other than the plot and log output widgets changes.
        """
        for name, widget in self.widgets.items():
            if name not in ["plot", "log_out"]:
                if isinstance(widget, widgets.Widget):
                    widget.observe(self._plot_episode, names="value")
                if isinstance(widget, dict):
                    for single_widget in widget.values():
                        single_widget.observe(self._plot_episode, names="value")
