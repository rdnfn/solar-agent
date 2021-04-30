"""Module with battery control environment of a photovoltaic installation."""

from typing import Tuple
import gym

from solara.environment.batteries import BatteryModel
from solara.environment.grids import GridModel
from solara.environment.loads import LoadModel
from solara.environment.photovoltaics import PVModel


class BatteryControlEnv(gym.Env):
    """A gym enviroment for controlling a battery in a PV installation."""

    # Set this in SOME subclasses
    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(
        self,
        battery: BatteryModel,
        pv_system: PVModel,
        grid: GridModel,
        load: LoadModel,
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
        raise NotImplementedError

    def reset(self) -> object:
        """Resets environment to initial state and returns an initial observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

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

    @property
    def unwrapped(self) -> None:
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self) -> str:
        """Return string."""
        if self.spec is None:
            return "<{} instance>".format(type(self).__name__)
        else:
            return "<{}<{}>>".format(type(self).__name__, self.spec.id)

    def __enter__(self) -> object:
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args) -> bool:
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False
