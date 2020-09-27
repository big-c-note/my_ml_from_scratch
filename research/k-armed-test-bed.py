from typing import Dict, List, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


if __name__ == "__main__":
    n_actions: int = 10
    epsilons: List[float] = [0, .001, .003, .01, .03, .1, .3]
    n_outer_iters: int = 2000
    n_inner_iters: int = 1000
    for epsilon in epsilons:
        avg_reward: np.ndarray = np.zeros(n_inner_iters,)
        for _ in trange(n_outer_iters):
            rewards: np.ndarray = np.random.normal(0, 1, n_actions)
            # The index will be the action chosen.
            q_actions: np.ndarray = np.zeros(n_actions,)
            num_actions: Dict[int, int] = {}
            for t in range(n_inner_iters):
                greedy_action: int = q_actions.argmax()
                # It's unlikely that the action-value estimates (Q(at)) would have
                # multiple tied for best. However, if they did, we may want to break
                # the tie at random. I believe argmax just takes the 1st index with the
                # max.
                if np.random.rand(1)[0] < epsilon:
                    explore_options: List[int] = [
                        x for x in range(10) if x != greedy_action
                    ]
                    a_t: int = np.random.choice(explore_options, 1)[0]
                else:
                    a_t = greedy_action
                r_t: float = np.random.normal(rewards[a_t], 1, 1)[0]
                # The reward comes from the normal distribution with expectation of the
                # randomly generated reward value and a sigma of 1.
                try:
                    num_actions[a_t] += 1
                except KeyError:
                    num_actions[a_t] = 1
                q_action: float = 1 / num_actions[a_t] * (r_t - q_actions[a_t])
                q_action = 1 / num_actions[a_t] * (r_t - q_actions[a_t])
                q_actions[a_t] += q_action
                avg_reward[t] += r_t / n_outer_iters

        rgb: Tuple[float, float, float] = (
            random.random(), random.random(), random.random()
        )
        plt.plot(
            np.array(list(range(n_inner_iters))),
            avg_reward,
            c=rgb,
            label=f"{epsilon}"
        )
    plt.legend(loc="upper left")
    plt.savefig('research/figures/k-armed-test-bed.png')
