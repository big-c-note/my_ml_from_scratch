from typing import Dict, List, Tuple, Optional
import random
import multiprocessing

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm


def _run_test(
    n_actions: int,
    n_outer_iters: int,
    n_inner_iters: int,
    exploration: str,
    c: Optional[int],
    epsilon: Optional[float],
    optimistic_value: Optional[int],
    optimistic_values: bool = False,
    stationary: bool = False,
    alpha: float = None
):
    avg_reward: np.ndarray = np.zeros(n_inner_iters,)
    optimal_actions: np.ndarray = np.zeros(n_inner_iters,)
    rewards: np.ndarray = np.random.normal(0, 1, n_actions)
    optimal_action: float = rewards.argmax()
    # The index will be the action chosen.
    q_actions: np.ndarray = np.zeros(n_actions,)
    if optimistic_values:
        q_actions = q_actions + optimistic_value
    num_actions: np.ndarray = np.zeros(n_actions,)
    for t in range(n_inner_iters):
        if not stationary:
            if np.random.rand(1)[0] > .98:
                rewards = np.random.normal(0, 1, n_actions)
                optimal_action = rewards.argmax()
        if exploration == 'epsilon-greedy':
            greedy_actions: np.array = np.argwhere(
                q_actions == np.max(q_actions)
            ).reshape(-1)
            greedy_action: int = np.random.choice(greedy_actions, 1)[0]
            # It's unlikely that the action-value estimates (Q(at)) would have
            # multiple tied for best. However, if they did, we may want to break
            # the tie at random. Of course, it would be more efficeint to
            # only do that if multiple actions are retuened.
            if np.random.rand(1)[0] < epsilon:
                explore_options: List[int] = [
                    x for x in range(10) if x != greedy_action
                ]
                a_t: int = np.random.choice(explore_options, 1)[0]
            else:
                a_t = greedy_action
        elif exploration == 'upper-confidence-bound':
            with np.errstate(all='ignore'):
                upper_bound_q_actions: np.ndarray = q_actions + (
                    c * np.sqrt(np.log(t + 1) / num_actions)
                )
            # Any nans are the result of a divide by 0. They need to be
            # explored ASAP.
            upper_bound_q_actions[np.isnan(upper_bound_q_actions)] = np.inf
            upper_bound_actions = np.argwhere(
                upper_bound_q_actions == np.max(upper_bound_q_actions)
            ).reshape(-1)
            a_t = np.random.choice(upper_bound_actions, 1)[0]
        if a_t == optimal_action:
            optimal_actions[t] += 1
        r_t: float = np.random.normal(rewards[a_t], 1, 1)[0]
        # The reward comes from the normal distribution with expectation of the
        # randomly generated reward value and a sigma of 1.
        try:
            num_actions[a_t] += 1
        except KeyError:
            num_actions[a_t] = 1
        if alpha:
            step_size: Optional[float] = alpha
        else:
            step_size = 1 / num_actions[a_t]
        q_action: float = step_size * (r_t - q_actions[a_t])
        q_action = step_size * (r_t - q_actions[a_t])
        q_actions[a_t] += q_action
        avg_reward[t] += r_t / n_outer_iters
    return avg_reward, optimal_actions


if __name__ == "__main__":
    num_cores: int = multiprocessing.cpu_count()
    n_actions: int = 10
    epsilons: List[float] = [.003, .01, .03, .1, .3, .5, .7]
    epsilon: Optional[float] = None
    exploration: str = 'upper-confidence-bound'
    cs: List[float] = [.5, 1, 1.5, 5, 7.5, 10]
    n_outer_iters: int = 2000
    n_inner_iters: int = 3000
    optimistic_values: bool = False
    optimistic_value: Optional[int] = None
    stationary: bool = True
    alpha: Optional[float] = None
    figure, axes = plt.subplots(nrows=2, ncols=1)
    for c in tqdm(cs):
        output_lst: List[np.ndarray] = Parallel(n_jobs=num_cores)(
            delayed(_run_test)(
                n_actions,
                n_outer_iters,
                n_inner_iters,
                exploration,
                c,
                epsilon,
                optimistic_value,
                optimistic_values,
                stationary,
                alpha
            ) for _ in range(n_outer_iters)
        )
        avg_reward: np.ndarray = sum(
            [pair[0] for pair in output_lst]
        )
        optimal_action: np.ndarray = sum(
            [pair[1] for pair in output_lst]
        ) / n_outer_iters
        rgb: Tuple[float, float, float] = (
            random.random(), random.random(), random.random()
        )
        axes[0].plot(
            np.array(list(range(n_inner_iters))),
            avg_reward,
            c=rgb,
            label=f"{c}"
        )
        axes[1].plot(
            np.array(list(range(n_inner_iters))),
            optimal_action,
            c=rgb,
            label=f"{epsilon}"
        )
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    figure.savefig('research/figures/k-armed-test-bed-non-stationary.png')
