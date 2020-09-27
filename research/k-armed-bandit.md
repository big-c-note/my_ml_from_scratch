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

The reward expectation for an action is defined as follows:

$$q_{*}(a) \doteq \mathbb{E}[R_{t} \vert A_{t} = a] $$

A few interesting parameters and situations arise. 

- You can choose the number of actions, $k$.
- For each a in A,$ you can choose the number of rewards, $R_{n}$.
- For each $a \in A$ you can choose a distribution of $R_t \vert A_t = a$ and the associated probability distribution. For now, we assume all time steps 
have the same probability distribution for each $R_t \vert A_t = a$.
    - The probability distribution for each reward can be known or can be figured from data.

Of course, having one reward per action would make finding the optimal action trivial.
You can still use a reinforcement learning system to learn that, if for example you don't know $q_{*}(a)$. 

In order to learn this, your reinforcement system cannot always choose the optimal action. Meaning, the 
system also has to _explore_ in order to find the optimal actions. Balancing _exploring_ and _exploiting_
(taking the _greedy_ action) is a delicate process. One needs to be careful that the method of doing so does not violate any conditions
surrounding stationarity, etc.

Here are a few simulations over epsilon greedy versus just greedy:
![K Armed Test Bed](research/figures/k-armed-test-bed.png)
