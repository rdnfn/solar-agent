"""Utility functions for RLlib."""
from __future__ import annotations

# Above enables using TYPE_CHECKING without using quotes around annotation

from typing import TYPE_CHECKING, Tuple, List, Dict

import numpy as np
import glob

if TYPE_CHECKING:
    import ray.rllib.agents.trainer


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
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)

    return (observations, actions, rewards, infos)


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
            check_save_path + "/checkpoint_0000{i:02.0f}/checkpoint-{i}".format(i=i)
        )
        observations, actions, rewards, infos = run_episode(agent)
        episode_dict = get_episode_dict(
            observations,
            actions,
            rewards,
            infos,
            all_obs_keys=agent.env_creator().obs_keys,
        )
        episode_dicts.append(episode_dict)

    return episode_dicts


def create_obs_dict(
    observations: List[np.array], obs_keys: List[str]
) -> Dict[str, np.array]:
    """Create a dictionary from observations returned during steps in Gym environment.

    Args:
        observations (List[np.array]): list of observations collected during episode
        obs_keys (List[str]): list of keys corresponding to each element in an
            observation after a single step in an environment. This may be
            obtainable via the `obs_keys` attribute of the env of the agent.

    Returns:
        Dict[str, np.array]: observations as a dictionary, with each key having the
            trajectory for a single observation type, e.g. battery content.
    """
    observations = np.array(observations)
    obs_dict = {}
    for i, key in enumerate(obs_keys):
        obs_dict[key] = observations[:, i]

    return obs_dict


def get_info(key: str, infos: List[Dict]) -> List:
    """Get a trajectory for a single key in a list of infos.

    This function combines the values stored in a list of info dictionaries returned
    whilst stepping through a Gym environment, and returns a trajectory for a single
    key in the info dictionaries as a list.

    Args:
        key (str): key in info dictionary.
        infos (List[Dict]): list of info dictionaries collected during episode.

    Returns:
        List: value trajectory over episode.
    """
    data = []
    for info in infos:
        data.append(info[key])

    return data


def get_episode_dict(
    observations: np.array,
    actions: List,
    rewards: List,
    infos,
    all_obs_keys: List[str],
    obs_keys: List[str] = None,
    info_keys: List[str] = None,
) -> Dict:
    """Get dictionary form of episode data.

    The return can be used for plotting an episode, and defines what is plotted
    for other functions.

    Args:
        observations (np.array): list of observations
        actions (List): list of actions
        rewards (List): list of rewards
        infos ([type]): list of infos
        all_obs_keys (List[str]): keys of values in observations
        obs_keys (List[str], optional): observations keys to be included in returned
            dict. Defaults to None.
        info_keys (List[str], optional): info keys to be included in returned dict.
            Defaults to None.

    Returns:
        Dict: dictionary used for plotting.
    """
    if obs_keys is None:
        obs_keys = ["load", "pv_gen", "energy_cont"]
    if info_keys is None:
        info_keys = ["net_load", "charging_power"]

    obs_dict = create_obs_dict(observations, all_obs_keys)
    episode_dict = dict((key, obs_dict[key]) for key in obs_keys if key in obs_dict)

    for key in info_keys:
        episode_dict[key] = get_info(key=key, infos=infos)

    episode_dict["rewards"] = rewards
    episode_dict["actions"] = actions

    return episode_dict
