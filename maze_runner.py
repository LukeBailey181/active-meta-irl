from maze_env import MutableMaze
import numpy as np
from manual_controller import ManualControl
from goal_setters import random_goal
from mazes import big_maze, small_maze
from helpers import *
import pygame

control_options = ['manual', 'random']
mode = 'manual'

# Define maze environment
env = MutableMaze(
    board_size=big_maze.shape[0],
    init_grid_string=big_maze,
    H=200,
    render_mode='human',)

# Take a random action at each step
if mode == 'random':
    while True:
        # action = env.action_space.sample()
        action = np.random.choice([0,1,2,3])
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            env.set_goal(random_goal(env))
            env.reset()
# Control using the arrow keys
elif mode == 'manual':
    manual_control = ManualControl(env, seed=42, set_goal=True)
    manual_control.start()
else:
    print("Not implemented")