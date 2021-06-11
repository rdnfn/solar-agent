"""Module that helps describe wired connections in environment."""

from __future__ import annotations

from typing import Union
import numpy as np


class PowerFlow:
    """Power flow."""

    def __init__(self, components: list(int), fully_connected=False) -> None:
        """Power flow.

        A data structure to describe connections and power flow in an electric system.

        Args:
            components (list): a list of components in the electric system
            fully_connected (bool, optional): whether all components should be
                connected. Defaults to False.
        """
        if len(components) > len(set(components)):
            raise ValueError(
                (
                    "Component list entries should be unique,"
                    " but for given list are not."
                )
            )

        self.components = components

        self.component_abbr = {
            component: str(component)[0:1] for component in components
        }

        num_comps = len(components)
        self.values = np.zeros((num_comps, num_comps), dtype=np.float32)

        if not fully_connected:
            self.connections = np.identity((num_comps, num_comps), dtype=bool)
        else:
            self.connections = np.ones((num_comps, num_comps), dtype=bool)

    def __getitem__(self, components: tuple(str)) -> float:
        """Get power flow of component, or between two components.

        Args:
            components (tuple(str)): selected component(s)

        Returns:
            float: power
        """
        source_idx, target_idx = self._get_idxs_from_components(components)
        return self.values[source_idx, target_idx]

    def __setitem__(self, components: Union[tuple(str), str], value):
        """Set power flow of component, or between two components.

        Args:
            components (Union[tuple): selected component(s)
            value ([type]): power value to set to
        """

        source_idx, target_idx = self._get_idxs_from_components(components)

        if not self.connections[source_idx, target_idx]:
            raise ValueError(
                (
                    "No connection between the given "
                    "components, thus no power flow possible."
                )
            )

        self.values[source_idx, target_idx] = value
        if source_idx != target_idx:
            self.values[target_idx, source_idx] = -value

    def _get_idxs_from_components(
        self, components: Union[tuple(str), str]
    ) -> tuple(float, float):
        """Get internal numeric indices for component(s).

        Args:
            components (Union[tuple, str]): component(s) to get indices for.

        Returns:
            float, float: numeric indices of components
        """

        if not isinstance(components, tuple):
            source_idx = self.index(components)
            target_idx = source_idx
        elif len(components) > 2:
            raise ValueError("components must either have length 1 or 2.")
        else:
            source_idx = self.index(components[0])
            target_idx = self.index(components[1])

        return source_idx, target_idx

    def index(self, component: str) -> int:
        """Get internal numeric index for component.

        Args:
            component (str): component name

        Returns:
            int: component index
        """
        return self.components.index(component)

    def add_connection(self, source: str, target: str) -> None:
        """Add connection between source and target components.

        Args:
            source (str): component
            target (str): component
        """

        source_idx = self.index(source)
        target_idx = self.index(target)

        self.connections[source_idx, target_idx] = True

    def get_connections(self) -> list(tuple(str, str)):
        """Get all (non-self-referenced) connections in electric system.

        Args:
            self ([type]): [description]

        Returns:
            list: [description]
        """
        connection_list = []
        connections_idxs = np.argwhere(self.connections is True)

        for source_idx, target_idx in connections_idxs:
            if source_idx != target_idx:
                source = self.components[source_idx]
                target = self.components[target_idx]
                connection_list.append((source, target))

        return connection_list
