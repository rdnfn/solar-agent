"""Utility functions for RLlib."""
from __future__ import annotations

# Above enables using TYPE_CHECKING without using quotes around annotation

from typing import TYPE_CHECKING, Tuple, List, Dict

import numpy as np
import glob

if TYPE_CHECKING:
    import ray.rllib.agents.trainer
    import gym


def run_episode(
    agent: ray.rllib.agents.trainer.Trainer, explore: bool = False
) -> Tuple:
    """Run an episode with an agent.

    This function runs an episode with an agent in the environment given by its
    `env_creator()` method.

    Args:
        agent (ray.rllib.agents.trainer.Trainer): agent to be used for episode.
        explore (bool): whether the agent should use exploration policy. Defaults to
            False.

    Returns:
        Tuple: observations, actions, rewards, info
    """

    done = False
    actions = []
    observations = []
    rewards = []
    infos = []

    env = agent.env_creator()
    obs = env.reset()
    observations.append(obs)

    # Running episode
    while not done:
        action = agent.compute_action(obs, explore=explore)
        obs, reward, done, info = env.step(action)
        actions.append(float(action))
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)

    return (observations, np.array(actions), np.array(rewards), infos)


def run_episodes_from_checkpoints(
    agent: ray.rllib.agents.trainer.Trainer, check_save_path: str
) -> List[Dict]:
    """Run episode from agent checkpoints and get corresponding episode trajectories.

    Args:
        agent (ray.rllib.agents.trainer.Trainer): agent to load checkpoints for.
        check_save_path (str): path where checkpoints are saved.

    Returns:
        List[Dict]: list of dictionaries, each with data from one episode.
    """

    final_iter_num = max(
        [int(dirname.split("_")[-1]) for dirname in glob.glob(check_save_path + "/*")]
    )

    episode_dicts = []

    for i in range(1, final_iter_num + 1):
        agent.restore(
            check_save_path + "/checkpoint_{i:06.0f}/checkpoint-{i}".format(i=i)
        )
        observations, actions, rewards, infos = run_episode(agent)
        episode_dict = get_episode_dict(
            observations,
            actions,
            rewards,
            infos,
        )
        episode_dicts.append(episode_dict)

    return episode_dicts


def concat_dict_data(dicts: List[Dict]) -> Dict[str, np.array]:
    """Concatenate list of dicts into dict of np.arrays.

    Each dictionary in list must have the same keys. For example, input
    `[{'a':1},{'a':2}]` is returned as `{'a': np.array([1,2])}`.

    Args:
        dicts (List[Dict]): list of dicts to be combined

    Returns:
        Dict[str, np.array]: dictionary with np.array values
    """
    concat_dict = {}
    for key in dicts[0].keys():
        concat_dict[key] = np.empty(len(dicts))

    for i, dictionary in enumerate(dicts):
        for key, value in dictionary.items():
            concat_dict[key][i] = value

    return concat_dict


def get_episode_dict(
    observations: List[Dict],
    actions: List,
    rewards: List,
    infos,
) -> Dict:
    """Get dictionary form of episode data.

    The return can be used for plotting an episode, and defines what is plotted
    for other functions.

    Args:
        observations (List): list of observations (of type gym.spaces.Dict)
        actions (List): list of actions
        rewards (List): list of rewards
        infos ([type]): list of infos

    Returns:
        Dict: dictionary used for plotting.
    """

    obs_dict = concat_dict_data(observations)
    info_dict = concat_dict_data(infos)

    episode_dict = {**obs_dict, **info_dict}

    episode_dict["rewards"] = rewards
    episode_dict["actions"] = actions

    return episode_dict


class DeterministicAgent:
    """Deterministic Agent."""

    def __init__(self, actions: List, env: gym.Env) -> None:
        """Deterministic Agent.

        Args:
            actions (List): list of actions the agent takes
            env (gym.Env): environment of the agent
        """
        self.actions = actions
        self.env = env
        self.step = 0

    def compute_action(
        self, obs: object, explore: bool = False
    ):  # pylint: disable=unused-argument
        """Get action.

        Args:
            obs (object): observations
            explore (bool, optional): Whether to explore, has no effect.
                Defaults to False.

        Returns:
            np.array: action taken by agent
        """

        action = self.actions[self.step]
        self.step += 1
        return action

    def env_creator(self) -> gym.Env:
        return self.env
