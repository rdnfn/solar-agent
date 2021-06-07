"""Module with functions to create envs from configs."""
from __future__ import annotations

from typing import TYPE_CHECKING
import solara.envs.components.solar
import solara.envs.components.load
import solara.envs.components.grid
import solara.envs.components.battery
import solara.envs.battery_control
from solara.envs.configs import DEFAULT_ENV_CONFIG

if TYPE_CHECKING:
    import gym


def create_env(env_config: dict = None) -> gym.Env:
    """Create a battery control environment from config.

    Args:
        env_config (Dict, optional): configuration dict for environment.
            Defaults to None.

    Returns:
        gym.Env: environment
    """

    if env_config is None:
        env_config = DEFAULT_ENV_CONFIG

    # Creating env components (battery, solar, etc.)
    components = {}
    for component in env_config["components"].keys():

        # Getting class
        class_name = env_config["components"][component].pop("type")
        component_module = getattr(solara.envs.components, component)
        component_class = getattr(component_module, class_name)

        # Creating component instance with config
        components[component] = component_class(**env_config["components"][component])

    # getting env class
    env_module_name, env_class_name = env_config["general"].pop("type").split(".")
    env_module = getattr(solara.envs, env_module_name)
    env_class = getattr(env_module, env_class_name)

    env = env_class(**env_config["general"], **components)

    return env
