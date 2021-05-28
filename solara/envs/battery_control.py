"""Module with battery control environment of a photovoltaic installation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import logging
import gym
import numpy as np

import solara.utils.logging

if TYPE_CHECKING:
    from solara.envs.components.battery import BatteryModel
    from solara.envs.components.grid import GridModel
    from solara.envs.components.load import LoadModel
    from solara.envs.components.solar import PVModel


class BatteryControlEnv(gym.Env):
    """A gym enviroment for controlling a battery in a PV installation."""

    def __init__(
        self,
        battery: BatteryModel,
        pv_system: PVModel,
        grid: GridModel,
        load: LoadModel,
        episode_len: float = 24,
        time_step_len: float = 1,
        grid_charging: bool = False,
        infeasible_control_penalty: bool = False,
        logging_level: str = "WARNING",
        log_handler: logging.Handler = None,
    ) -> None:
        """A gym enviroment for controlling a battery in a PV installation.

        This class inherits from the main OpenAI Gym class. The initial non-implemented
        skeleton methods are copied from the original gym class:
        https://github.com/openai/gym/blob/master/gym/core.py

        The main API methods that users of this class need to know are:
            step
            reset
            render
            close
            seed
        And set the following attributes:
            action_space: The Space object corresponding to valid actions
            observation_space: The Space object corresponding to valid observations
            reward_range: A tuple corresponding to the min and max possible rewards
        Note: a default reward range set to [-inf,+inf] already exists. Set it if you
        want a narrower range. The methods are accessed publicly as "step", "reset",
        etc...
        """

        self.battery = battery
        self.pv_system = pv_system
        self.grid = grid
        self.load = load
        self.components = [battery, pv_system, grid, load]

        self.episode_len = episode_len
        self.time_step_len = time_step_len
        self.grid_charging = grid_charging
        self.infeasible_control_penalty = infeasible_control_penalty

        # Setting up action and observation space

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        spaces = {
            "load": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "battery_cont": gym.spaces.Box(
                low=0, high=self.battery.size, shape=(1,), dtype=np.float32
            ),
            "time_step": gym.spaces.Discrete(self.episode_len + 1),
            "cum_load": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "cum_pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
        }
        self.observation_space = gym.spaces.Dict(spaces)

        (
            self.min_charge_power,
            self.max_charge_power,
        ) = self.battery.get_charging_limits()

        self.obs_keys = [
            "load",
            "pv_gen",
            "battery_cont",
            "time_step",
            "cum_load",
            "cum_pv_gen",
        ]

        self._setup_logging(logging_level, log_handler)
        self.logger.info("Environment initialised.")

        self.reset()

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step()
                calls will return undefined results
            info (dict): contains auxiliary diagnostic information
                (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        action = float(action)  # getting the float value
        self.logger.debug("step - action: %1.3f", action)

        # Get old state
        load, pv_generation, _, _, sum_load, sum_pv_gen = self.state.values()

        # Actions are proportion of max/min charging power, hence scale up
        if action > 0:
            action *= self.max_charge_power
        else:
            action *= -self.min_charge_power

        attempted_action = action.copy()

        if not self.grid_charging:
            # If charging from grid not enabled, limit charging to solar generation
            action = np.minimum(action, pv_generation)

        charging_power = self.battery.charge(power=action)

        # Get the net load after accounting for power stream of battery and PV
        net_load = load + charging_power - pv_generation
        net_load = np.maximum(net_load, 0)

        self.logger.debug("step - net load: %s", net_load)

        # Draw remaining net load from grid and get price paid
        cost = self.grid.draw_power(power=net_load)

        reward = -cost

        # Add impossible control penalty to cost
        if self.infeasible_control_penalty:
            power_diff = np.abs(charging_power - float(attempted_action))
            reward -= power_diff
            self.logger.debug("step - cost: %6.3f, power_diff: %6.3f", cost, power_diff)

        # Get load and PV generation for next time step
        load = self.load.get_next_load()
        pv_generation = self.pv_system.get_next_generation()

        sum_load += load
        sum_pv_gen += pv_generation
        self.time_step += 1

        observation = {
            "load": np.array([load], dtype=np.float32),
            "pv_gen": np.array([pv_generation], dtype=np.float32),
            "battery_cont": np.array(
                [self.battery.get_energy_content()], dtype=np.float32
            ),
            "time_step": int(self.time_step),
            "cum_load": sum_load,
            "cum_pv_gen": sum_pv_gen,
        }

        self.state = observation

        done = self.time_step >= self.episode_len

        info = {
            "net_load": net_load,
            "charging_power": charging_power,
            "cost": cost,
            "power_diff": power_diff,
        }
        info = {**info, **self.grid.get_info()}

        self.logger.debug("step - info %s", info)

        self.logger.debug(
            "step return: obs: %s, rew: %6.3f, done: %s", observation, reward, done
        )

        return (observation, float(reward), done, info)

    def reset(self) -> object:
        """Resets environment to initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        self.state = {
            "load": np.array([0.0], dtype=np.float32),
            "pv_gen": np.array([0.0], dtype=np.float32),
            "battery_cont": np.array([0.0], dtype=np.float32),
            "time_step": 0,
            "cum_load": np.array([0.0], dtype=np.float32),
            "cum_pv_gen": np.array([0.0], dtype=np.float32),
        }
        self.time_step = np.array([0])

        self.battery.reset()
        self.load.reset()
        self.pv_system.reset()

        self.logger.debug("Environment reset.")

        return self.state

    def render(self, mode: str = "human") -> None:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self) -> None:
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed: int = None) -> None:
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        raise NotImplementedError

    def _setup_logging(self, logging_level: str, log_handler: str = None) -> None:
        """Setup logger and handler."""

        if logging_level == "RAY":
            logging_level = logging.getLogger("ray.rllib").level

        solara.utils.logging.setup_log_print_options()

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging_level)

        if log_handler is None:
            self.log_handler = solara.utils.logging.OutputWidgetHandler()
        else:
            self.log_handler = log_handler
        self.logger.addHandler(self.log_handler)

        for component in self.components:
            component.set_log_handler(self.log_handler)
            component.set_log_level(logging_level)
