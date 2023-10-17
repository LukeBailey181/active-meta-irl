import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from maze_env import MutableMaze, Trajectory
import matplotlib.pyplot as plt
from goal_setters import random_goal
from itertools import product               # Cartesian product for iterators
from mazes import *
from helpers import ExpSga, Constant, linear_decay


# Maxent IRL implementation taken from 
# Harvard CS282 fall 2023 Homework 2

def compute_expected_svf(p_initial, terminal, reward, env, eps=1e-5):

    n_states = env.get_num_states()
    n_actions = 4
    #n_states, _, n_actions = p_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)                             # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    print("DEBUG")
    print(n_states)
    for _ in range(2 * n_states):                       # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))            # za: action partition function

        print("CHECKING RUN")
        print("_")
        # for each state-action pair
        #for s_from, a in product(range(n_states), range(n_actions)):
        #    # sum over s_to
        #    for s_to in range(n_states):
        #        #za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]
        #        za[s_from, a] += env.T_func_index(s_from, s_to, a) * np.exp(reward[s_from]) * zs[s_to]
        #        #za[s_from, a] += p_transition(s_from_state, a, s_to_state) * np.exp(reward[s_from]) * zs[s_to]

        # ====== Vectorized version ======
        # Expand dimensions to facilitate broadcasting
        expanded_reward = np.exp(reward)[:, np.newaxis, np.newaxis]
        zs_expanded = zs[np.newaxis, np.newaxis, :]

        # Compute the product
        product = env.T_matrix * expanded_reward * zs_expanded

        # Sum over s_to (axis=2)
        za = np.sum(product, axis=2)
        # ================================

        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))              # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):                    # longest trajectory: n_states

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                #d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * p_transition[s_from, s_to, a]
                d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * env.T_func_index(s_from, s_to, a)

    # 6. sum-up frequencies
    return d.sum(axis=1)


def feature_expectation_from_trajectories(features, trajectories, env):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            fe += features[env.state_to_index(s), :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

def initial_probabilities_from_trajectories(n_states, trajectories, env):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        starting_state = t.transitions()[0][0]
        index = env.state_to_index(starting_state)
        p[index] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize

def maxent_irl(features, terminal, trajectories, optim, init, env, eps=1e-4):
    n_states = env.get_num_states()
    n_actions = 4
    #n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    print("got here 1")
    e_features = feature_expectation_from_trajectories(features, trajectories, env)

    print("got here 2")
    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories, env)

    # gradient descent optimization
    omega = init(n_features)        # initialize our parameters
    delta = np.inf                  # initialize delta for convergence check

    optim.reset(omega)              # re-start optimizer

    i = 0
    print("got here 3")
    while delta > eps:              # iterate until convergence
        print(i)
        print(delta)
        i += 1 
        if i == 10000: 
            print(delta)
            i = 0

        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        print("got here 4")
        e_svf = compute_expected_svf(p_initial, terminal, reward, env)
        grad = e_features - features.T.dot(e_svf)

        print("got here 5")
        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega)
 
def MaxEnt(
    train_trajectories,
    env,
    test_dataset=None,
    epochs=2000,
    epsilon=0.1,
    batch_size=32,
    lr=0.001,
    eval_freq=None,
    save_weights=False,
    r="",
):

    # choose our parameter initialization strategy:
    # initialize parameters with constant
    init = Constant(1.0)

    # choose our optimization strategy:
    # we select exponentiated stochastic gradient descent with linear learning-rate decay
    optim = ExpSga(lr=linear_decay(lr0=0.2))

    features = env.get_features()

    N = env.board_size
    terminal = [env.state_to_index((x, y, x, y)) for x in range(N) for y in range(N)]

    reward_maxent = maxent_irl(features, terminal, train_trajectories, optim, init, env)
    
    

