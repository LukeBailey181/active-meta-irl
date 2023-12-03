import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from maze_env import MutableMaze
import matplotlib.pyplot as plt
from goal_setters import random_goal
from mazes import *
import os
import yaml
from helpers import Net, ConvNet
from maze_env import Trajectory

def goal_agent_manhattan(m1, m2):
    agent_1 = np.argwhere(m1 == 2)[0]
    agent_2 = np.argwhere(m2 == 2)[0]

    goal_1 = np.argwhere(m1 == 3)[0]
    goal_2 = np.argwhere(m2 == 3)[0]

    m1_blocks = m1 == 1
    m2_blocks = m2 == 1

    weights = [1, 1, 1]

    N = m1_blocks.shape[0]

    # Compute the manhattan distance between the two agents
    agent_manhattan = (
        np.abs(agent_1[0] - agent_2[0]) + np.abs(agent_1[1] - agent_2[1])
    ) / (2 * N)

    # Compute the manhattan distance between the two goals
    goal_manhattan = (np.abs(goal_1[0] - goal_2[0]) + np.abs(goal_1[1] - goal_2[1])) / (
        2 * N
    )

    overlap = np.sum(m1_blocks != m2_blocks) / N**2

    heuristic = (
        weights[0] * agent_manhattan
        + weights[1] * goal_manhattan
        + weights[2] * overlap
    )
    # Turn heuristic from a numpy array into a float

    return heuristic

def goal_manhattan(m1, m2):

    goal_1 = np.argwhere(m1 == 3)[0]
    goal_2 = np.argwhere(m2 == 3)[0]

    m1_blocks = m1 == 1

    N = m1_blocks.shape[0]

    # Compute the manhattan distance between the two goals
    goal_manhattan = (np.abs(goal_1[0] - goal_2[0]) + np.abs(goal_1[1] - goal_2[1])) / (
        2 * N
    )


    return goal_manhattan

def edit_distance(m1, m2):
    """Compute the edit distance between two mazes"""
    return (m1 != m2).sum()
