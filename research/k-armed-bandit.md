---
header-includes:
    - \usepackage[]{algorithm2e}
title: K Armed Bandit Notes
author: Colin Manko^1^\footnote{Corresponding author – colin@colinmanko.com}
date: \today
abstract: Quick overview of the neccessary formulas and concepts for K Armed Bandits. I'm reading Reinforcement Learning by Sutton and Barton.

---
# Introduction

The K Armed Bandit can model a situation where an agent faces $k$ actions, $a \in A$, each with 
a distribution of possible rewards, $r \in R$, and their corresponding probabilities of occurring, 
$p(r_{t} \vert a_{t} = a)$. The goal of this situation is to maximize the summation of rewards 
through time step $T$, $\sum_{t=1}^{T}R_{t}$. We do not transition to another state after selecting 
an action. Instead, we repeatedly face the same situation $T$ times and we keep track of $a_{n}$, 
the number of times action $a$ was selected. As ${a_{n}\to\infty}$, we can cacluate an accurate
reward expectation, this assumes the probability ditributions over $r \in R \vert A$ are stationary, or non-changing. 
In truth, stationary refers more to having a non-changing $q(*)$, or the true expectation. Theoretically, changing the
probabilities associated with the rewards for a given action could result in the same $q(*)$.

The reward expectation for an action is defined as follows:

$$q_{*}(a) \doteq \mathbb{E}[R_{t} \vert A_{t} = a] $$

## Incremental Updates.
$Q_n$ refers to the value estimate after $n$ occurances of $a$.

$Q_n = \frac{R_1 + R_2 + R_3 + \ldots + R_{n-1}}{n-1}$
$Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n}R_i$
$Q_{n+1} = \frac{1}{n}(R_n + \sum_{i=1}^{n-1}R_i)$
$Q_{n+1} = \frac{1}{n}(R_n + (n-1)\times\frac{1}{n-1}\sum_{i=1}^{n-1}R_i)$
$Q_{n+1} = \frac{1}{n}(R_n + (n-1)Q_n)$
$Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$

Instead of the $\frac{1}{n}$, we can use a constant step size parameter named $\alpha$. This means that 
the step size is non-decreasing. Whereas, the above is good for stationary problems (due to convergence properties, 
the constant step size can be good for non-stationary problems, although it is not gauranteed to converge

## A few interesting parameters and situations arise. 

- You can choose the number of actions, $k$.
- For each a in A,$ you can choose the number of rewards, $R_{n}$.
- For each $a \in A$ you can choose a distribution of $R_t \vert A_t = a$ and the associated probability distribution. For now, we assume all time steps 
have the same probability distribution for each $R_t \vert A_t = a$.
    - The probability distribution for each reward can be known or can be figured from data.
- These reward distributions, per each action, could be stationary or non-stationary.
- Can have different action-value estimation methods. The sample-average method is the one in the formula
listed above, which is good for stationary problems, but not non-stationary problems where you may want to 
not decrease the step-size parameter, as to give more weight to recent values.
- The number of time steps, and whether there are continuous or episodic (this is not as big of a deal in k armed bandit,
but it is helpful to start thinking about this.)
- Changing the amount of exploration over time.

Of course, having one reward per action would make finding the optimal action trivial.
You can still use a reinforcement learning system to learn that, if for example you don't know $q_{*}(a)$. 

## K Armed Test Bed.

To give some figures to the exploration of these concepts, I have implemented my own K armed bandit test bed.
In all, there are 1000 iterations 2000 times. I may change the distributions of rewards and whether those distributions
are stationary or non-stationary. I'll specifiy for each graph discussed. 

## Exploration.
In order to learn this, your reinforcement system cannot always choose the optimal action. Meaning, the 
system also has to _explore_ in order to find the optimal actions. Balancing _exploring_ and _exploiting_
(taking the _greedy_ action) is a delicate process. One needs to be careful that the method of doing so does not violate any conditions
surrounding stationarity, etc. Also, given the dynamics of the systems (as detailed in the bullet points above), there could
be an optimal configuration of _exploration_ and _greedy_ behavior.

Here are a few simulations over epsilon greedy versus just greedy:
![K Armed Test Bed](research/figures/k-armed-test-bed.png)

The above k armed bandit had 10 actions with rewards randomly generated from a normal distribution with variance 1 and mean 0. Once found,
the reward distributions were given by a normal distribution with variance 1 and mean of the given reward. We find that there is a trade off
between the number of iterations and the amount of exploring. Over a long enough timeline it seems as though a small value for exploration 
would suffice. It is possible to change the amount of exploration over time. It is also possible to optimize uncertainty by using an upperbound of uncertainty, to
prioritize actions that have high uncertainty (or really prioritized by chance of being the optimal value.). Of course, this depends on how often
and for how long the dynamics change and for how many time steps you expect to train, etc.

A note about variance. If variance was 0, you would want a greedy method. A variance of zero means that the rewards are deterministic and simply
choosing the highest will result in the optimal value add.

A note about stationarity, if the dynamics were non-stationary, you would want more exploration to pick up on the changing
dynamics.

I should also note that when you are exploring, you should choose randomly to break ties with an argmax.
