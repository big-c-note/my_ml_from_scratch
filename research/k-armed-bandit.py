from typing import Dict, List

import numpy as np


# Pg. 25-36
# K Armed Bandit
# Your faced with the same choice over and over.
# You have k actions to choose from, each of which garner a reward from a
# stationary (unchanging) probability distribution.
# Goal: Over some number of time steps you hope maximize reward.
def make_actions(
    n_actions: int,
    probabilities: np.ndarray
) -> Dict[int, float]:
    """
    Create a dictionary of actions and associated probabilities.

    Parameters
    ----------
    n_actions : int
        The number of actions.
    probabilities : np.ndarray
        The probabilities associated with the actions.

    Returns
    -------
    actions : Dict[int, float]
        Dictionary of actions and associated probabilities.
    """
    assert len(probabilities) == n_actions
    assert np.isclose(np.sum(probabilities), 1, rtol=.01)
    actions: Dict[int, float] = {}
    for i, prob in enumerate(probabilities):
        actions[i] = prob
    return actions


def make_rewards(
    n_actions: int,
    rewards: np.ndarray
) -> Dict[int, float]:
    """
    Create a dictionary of actions and associated rewards.

    Parameters
    ----------
    n_actions : int
        The number of actions.
    rewards : np.ndarray
        The associated rewards with the actions.

    Returns
    -------
    rewards_dict : Dict[int, float]
        Dictionary of actions and associated rewards.
    """
    assert len(rewards) == n_actions
    rewards_dict: Dict[int, float] = {}
    for i, reward in enumerate(rewards):
        rewards_dict[i] = reward
    return rewards_dict


if __name__ == "__main__":
    # Trivial example if the action-value comes from a probability distribution
    # that is just 1 (ie; certain that one value happens)
    n_actions: int = 5
    unnormalized_probs: np.ndarray = np.random.rand(n_actions)
    probabilities: np.ndarray = unnormalized_probs / np.sum(unnormalized_probs)
    rewards: np.ndarray = np.random.rand(n_actions)
    actions: Dict[int, float] = make_actions(n_actions, probabilities)
    rewards_dict: Dict[int, float] = make_rewards(n_actions, rewards)
    num_iterations: int = 1000
    q_stars: Dict[int, float] = {}
    num_actions: Dict[int, int] = {}
    for i in range(1000):
        probs: List[float] = list(actions.values())
        a_t: int = np.random.choice(list(actions.keys()), 1, probs)[0]
        try:
            num_actions[a_t] += 1
        except KeyError:
            num_actions[a_t] = 1
        try:
            q_stars[a_t] += 1 / num_actions[a_t] * (rewards[a_t] - q_stars[a_t])
        except KeyError:
            q_stars[a_t] = 0
            q_stars[a_t] += 1 / num_actions[a_t] * (rewards[a_t] - q_stars[a_t])
    import ipdb;
    ipdb.set_trace()
