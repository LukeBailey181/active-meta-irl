import numpy as np
from goal_setters import random_goal
from helpers import solve_maze, generate_maze
from maze_env import Trajectory
from helpers import generateExpertTrajectory
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Aquisition functions
def failed_min_max_acquisition(
        failed_configs, 
        cur_mazes, 
        num_trajectories,
        env,
        config,
):
    """
    Args: 
        failed_configs: configurations of the mazes failed by last round of BC 
        cur_mazes: a list of the mazes that the current training set contains trajectories from.
        heurisitc: a function that takes in two mazes and returns a heuristic value
        num_trajectories: the number of trajectories to query from the expert
        env: environment to query the expert in
        config: experiment configuration
    
    Returns:
        states: a list of states queried from the expert
        actions: a list of actions queries from the expert
        cur_mazes: a list of the mazes that the current training set contains trajectories from.
        info: a dictionary containing non-essential information
    """
    
    if num_trajectories == 0:
        return [], [], cur_mazes, {}

    states = []     # List of states returned by aquisition function
    actions = []    # List of actions returned by aquisition function
    r = config["base"]["randomize"]
    heuristic = config["al"]["heuristic"]

    max_heuristic = 0
    if len(failed_configs) != 0:
        heuristics = np.zeros(len(failed_configs))
        max_heuristic = 0
        max_heuristic_idx = 0
        for t in range(len(failed_configs)):
            cur_maze_heuristics = np.zeros(len(cur_mazes))
            maze_count = 0
            for maze in cur_mazes:
                cur_maze_heuristics[maze_count] = heuristic(maze, failed_configs[t])
                maze_count += 1
            min_heuristic = np.min(cur_maze_heuristics)

            heuristics[t] = min_heuristic

        max_heuristic = np.max(heuristics)

        print(f"Max heuristic: {max_heuristic}")

        #num_left = min(batch_size, budget - len(trajectories))
        num_avail = len(failed_configs)

        if num_avail >= num_trajectories:
            # Find the num_left best heuristics
            best_heuristics = np.argsort(heuristics)[-num_trajectories:]
            # Append the corresponding configs to mazes
            for h in best_heuristics:
                cur_mazes.append(np.copy(failed_configs[h]))
                trajectory = generateExpertTrajectory(
                    env, r=config["base"]["randomize"], maze=failed_configs[h]
                )
                for state, action, _ in trajectory.transitions():
                    states.append(state)
                    actions.append(action)
        else:
            # Append all of the configs to mazes
            for h in range(num_avail):
                cur_mazes.append(np.copy(failed_configs[h]))
                trajectory = generateExpertTrajectory(
                    env, r=config["base"]["randomize"], maze=failed_configs[h]
                )
                for state, action, _ in trajectory.transitions():
                    states.append(state)
                    actions.append(action)
            # Append random mazes to mazes
            for h in range(num_trajectories - num_avail):
                if r == "m":
                    cur_mazes.append(np.copy(generate_maze(env.board_size)))
                elif r == "g":
                    goal_location = random_goal(env)
                    env.set_goal(goal_location)
                    cur_mazes.append(np.copy(env.grid_string))


                trajectory = generateExpertTrajectory(
                    env, r=config["base"]["randomize"], maze=cur_mazes[-1]
                )
                for state, action, _ in trajectory.transitions():
                    states.append(state)
                    actions.append(action)
    else:
        # Append random mazes to mazes
        #num_left = min(B, budget - len(trajectories))
        for h in range(num_trajectories):
            cur_mazes.append(np.copy(generate_maze(env.board_size)))
            trajectory = generateExpertTrajectory(
                env, r=config["base"]["randomize"], maze=cur_mazes[-1]
            )
            for state, action, _ in trajectory.transitions():
                states.append(state)
                actions.append(action)

    # Any useful information to keep track of
    info = {
        "max_heuristic": max_heuristic,
    }

    return states, actions, cur_mazes, info


# Aquisition functions
def new_acquisition(
        failed_configs, 
        cur_mazes, 
        num_trajectories,
        env,
        config,
):
    """
    Args: 
        failed_configs: configurations of the mazes failed by last round of BC 
        cur_mazes: a list of the mazes that the current training set contains trajectories from.
        heurisitc: a function that takes in two mazes and returns a heuristic value
        num_trajectories: the number of trajectories to query from the expert
        env: environment to query the expert in
        config: experiment configuration
    
    Returns:
        states: a list of states queried from the expert
        actions: a list of actions queries from the expert
        cur_mazes: a list of the mazes that the current training set contains trajectories from.
        info: a dictionary containing non-essential information
    """

    return states, actions, cur_mazes, info