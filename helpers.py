import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.colors import ListedColormap
import torch.nn as nn
import torch.nn.functional as F
import torch
import PIL.Image as Image
from maze_env import Trajectory
from goal_setters import random_goal

#################################################################################
###  This file contains a number of helper functions, including:              ###
###                                                                           ###
###   key_to_action: a dictionary mapping strings to actions                  ###
###   action_to_key: a dictionary mapping actions to strings                  ###
###                                                                           ###
###   get_transition_matrix: a function given by                              ###
###                grid_string (N, N) ---> Transition Matrix (N^2, A, N^2)    ###
###   visualize_transition: a function given by                               ###
###                Transition Matrix (N^2, A, N^2) X State (x,y) ---> Heatmap ###
###   solve_maze: a function given by                                         ###
###                grid_string (N, N) X  ---> Optimal Moves (N, N)            ###
###   visualize_optimal_moves: a function given by                            ###
###                Optimal Moves (N, N) ---> Heatmap + Arrows of the maze     ###
#################################################################################

maze_map = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])

# Dictionaries with all the possible actions/keys/tuples
key_to_action = {
    "left": 2,
    "right": 0,
    "up": 3,
    "down": 1,
}

action_to_key = {
    2: "left",
    0: "right",
    3: "up",
    1: "down",
}

tuple_to_key = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


# Define the network
class Net(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, state_size):
        self.fc_size = 32 * state_size**2
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class ConvNetMC(nn.Module):
    def __init__(self, state_size):
        self.fc_size = 32 * state_size**2
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


# Returns the transition matrix for a given maze
#     T[i * board_size + j][action][k * board_size + l] = P((k,l) | (i,j), action)
def get_transition_matrix(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N**2, 4, N**2))
    grid_string = grid_string.T

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i, j] == 1 or grid_string[i, j] == 3:
                # print(i * N + j)
                T[i * N + j][:, i * N + j] = 1
                continue

            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    # print(i, j)
                    if grid_string[i, j + 1] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, i * N + j + 1] = 1
                elif action == 1:
                    if grid_string[i + 1, j] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, (i + 1) * N + j] = 1
                elif action == 2:
                    if grid_string[i, j - 1] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, i * N + j - 1] = 1
                elif action == 3:
                    if grid_string[i - 1, j] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, (i - 1) * N + j] = 1
    return T


def get_transition_deltas(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N, N, 4, 2))
    # grid_string = grid_string.T

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i, j] == 1 or grid_string[i, j] == 3:
                # print(i * N + j)
                # T[i * N + j][:, i * N + j] = 1
                T[i, j, :, 0] = i
                T[i, j, :, 1] = j
                continue

            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    # print(i, j)
                    if grid_string[i, j + 1] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = 0
                    else:
                        # T[i * N + j, action, i * N + j + 1] = 1
                        T[i, j, action, 0] = 1
                        T[i, j, action, 1] = 0
                elif action == 1:
                    if grid_string[i + 1, j] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = 0
                    else:
                        # T[i * N + j, action, (i + 1) * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = 1
                elif action == 2:
                    if grid_string[i, j - 1] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = 0
                    else:
                        # T[i * N + j, action, i * N + j - 1] = 1
                        T[i, j, action, 0] = -1
                        T[i, j, action, 1] = 0
                elif action == 3:
                    if grid_string[i - 1, j] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = 0
                    else:
                        # T[i * N + j, action, (i - 1) * N + j] = 1
                        T[i, j, action, 0] = 0
                        T[i, j, action, 1] = -1
    return np.transpose(T, axes=[1, 0, 2, 3])


def get_transition_states(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N, N, 4, 2))
    # grid_string = grid_string.T

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i, j] == 1 or grid_string[i, j] == 3:
                # print(i * N + j)
                # T[i * N + j][:, i * N + j] = 1
                T[i, j, :, 0] = i
                T[i, j, :, 1] = j
                continue

            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    # print(i, j)
                    if grid_string[i, j + 1] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i
                    else:
                        # T[i * N + j, action, i * N + j + 1] = 1
                        T[i, j, action, 0] = j + 1
                        T[i, j, action, 1] = i
                elif action == 1:
                    if grid_string[i + 1, j] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i
                    else:
                        # T[i * N + j, action, (i + 1) * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i + 1
                elif action == 2:
                    if grid_string[i, j - 1] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i
                    else:
                        # T[i * N + j, action, i * N + j - 1] = 1
                        T[i, j, action, 0] = j - 1
                        T[i, j, action, 1] = i
                elif action == 3:
                    if grid_string[i - 1, j] == 1:
                        # T[i * N + j, action, i * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i
                    else:
                        # T[i * N + j, action, (i - 1) * N + j] = 1
                        T[i, j, action, 0] = j
                        T[i, j, action, 1] = i - 1
    return np.transpose(T, axes=[1, 0, 2, 3])


# Shows the transition probabilities for a given state given transition matrix T
def visualize_transition(T, state):
    N = int(np.sqrt(T.shape[0]))
    x, y = state
    fig, axs = plt.subplots(2, 2)

    # For each action
    for i in range(4):
        t_grid = T[x * N + y][i].reshape((N, N))
        # Plot the transition probabilities as a heatmap
        axs[i // 2, i % 2].imshow(t_grid, cmap="hot", interpolation="nearest")
        axs[i // 2, i % 2].set_title(f"Action {action_to_key[i]}")
        axs[i // 2, i % 2].set_xlabel("x")
        axs[i // 2, i % 2].set_ylabel("y")

        # add a red dot to the current state
        axs[i // 2, i % 2].scatter(y, x, c="r", s=40)

    plt.show()


def visualize_reward(maze, reward):
    # Create two subplots size by side
    fig, axs = plt.subplots(1, 2)
    # On the left subplot plot the maze
    cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])

    # Visualize the maze
    axs[0].imshow(maze.T, cmap=cmap, interpolation="nearest")
    axs[0].set_title("Maze")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    x = np.arange(0, maze.shape[0])
    y = np.arange(0, maze.shape[1])

    walls = maze == 1

    axs[1].imshow(walls.T, cmap="binary", interpolation="nearest")

    reward[walls == 1] = np.nan

    # On the right subplot plot the reward
    axs[1].pcolormesh(x, y, reward.T)
    # Make the walls black
    # axs[1].imshow(maze.T, cmap=cmap, interpolation="nearest", alpha=0.3)
    axs[1].set_title("Reward")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Return the figure
    # plt.show()
    return fig


# Returns a matrix of optimal moves for a given maze at each state
def solve_maze(maze):
    maze = maze.T
    goal = np.argwhere(maze == 3)[0]
    maze = (maze == 1).astype(int)

    optimal_moves = np.zeros(maze.shape, dtype=int) - 1
    # The moves at the goal + walls don't matter
    optimal_moves[goal[0]][goal[1]] = key_to_action["right"]
    optimal_moves[maze == 1] = 0

    # Work back through all squares to get the optimal moves
    current = deque([goal])

    while current:
        x, y = current.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Check if the neighbor is valid
            new_x, new_y = x + dx, y + dy
            if (
                0 <= new_x < maze.shape[0]
                and 0 <= new_y < maze.shape[1]
                and optimal_moves[new_x][new_y] == -1
            ):
                # If so, add it to the search queue and set its optimal move
                current.append((new_x, new_y))
                optimal_moves[new_x][new_y] = key_to_action[tuple_to_key[(-dx, -dy)]]

    # return optimal_moves
    return optimal_moves.T


# Visualizes the optimal moves for a given maze
def visualize_optimal_moves(maze, optimal_moves, save=False):
    # Plot the maze so that walls are black, empty spaces are white, the goal is green, and the start is blue
    cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])
    # maze = maze.T

    # Visualize the maze
    plt.imshow(maze.T, cmap=cmap, interpolation="nearest")
    plt.title("Optimal Maze Actions")
    N = maze.shape[0]

    # Parameters for the arrows
    hw = 0.5
    hl = 0.55
    so = 0.2
    l = 0.005
    col = "red"

    # Visualize the optimal moves
    for i in range(N):
        for j in range(N):
            if maze[i][j] == 1:
                continue
            if optimal_moves[i][j] == key_to_action["left"]:
                plt.arrow(
                    i + so, j, -l, 0, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["right"]:
                plt.arrow(
                    i - so, j, l, 0, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["up"]:
                plt.arrow(
                    i, j + so, 0, -l, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["down"]:
                plt.arrow(
                    i, j - so, 0, l, head_width=hw, head_length=hl, fc=col, ec=col
                )
    # plt.show()
    # Save the image
    if save:
        plt.savefig("optimal_directions.png")
    else:
        plt.show()

def generateExpertTrajectory(env, r="", maze=None):
    if maze is not None:
        obs = env.reset(grid_string=maze)
        obs_s = env.get_state_obs()
    elif r == "g":
        env.set_goal(random_goal(env))
        obs = env.reset()
        obs_s = env.get_state_obs()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
        obs_s = env.get_state_obs()
    else:
        obs = env.reset()
        obs_s = env.get_state_obs()

    policy = solve_maze(env.grid_string)

    cur_trajectory = Trajectory()
    while True:
        action = policy[obs_s[0], obs_s[1]]

        obs_old = obs

        obs, reward, term, trunc, info = env.step(action)
        obs_s = env.get_state_obs()

        cur_trajectory.add_transition(list(obs_old), action, obs)

        env.render()

        if term or trunc:
            break

    return cur_trajectory


def generate_maze(n):
    # Generate a (possibly impossible) maze
    def _gen_try(n):
        maze = np.ones((n, n), dtype=int)
        maze[1:-1, 1:-1] = 0
        for i in range(2, n - 2, 2):
            for j in range(2, n - 2, 2):
                maze[i, j] = 1
                if i == 2:
                    maze[i - 1, j] = 1
                if j == n - 3:
                    maze[i, j + 1] = 3
                if np.random.randint(0, 2) == 0:
                    maze[i + 1, j] = 1
                else:
                    maze[i, j + 1] = 1
        return maze

    maze = _gen_try(n)

    # Set a random goal
    zero_idx = np.argwhere(maze == 0)
    goal_idx = np.random.randint(0, len(zero_idx))
    maze[zero_idx[goal_idx][0], zero_idx[goal_idx][1]] = 3

    # Solve the maze
    solution = solve_maze(maze)

    # Check the number of squares unreachable from the goal against the total number of free squares
    num_dud_squares = np.sum(solution == -1)
    num_free_squares = np.sum(maze == 0)

    # Repeat until the goal is reachable from at least 90% of the free squares
    while num_dud_squares / num_free_squares > 0.1:
        maze = _gen_try(n)
        zero_idx = np.argwhere(maze == 0)
        goal_idx = np.random.randint(0, len(zero_idx))
        maze[zero_idx[goal_idx][0], zero_idx[goal_idx][1]] = 3
        solution = solve_maze(maze)
        num_dud_squares = np.sum(solution == -1)
        num_free_squares = np.sum(maze == 0)

    # Turn all dud squares into walls
    maze[solution == -1] = 1

    zero_idx = np.argwhere(maze == 0)

    # Add a random start position
    start_idx = np.random.randint(0, len(zero_idx))
    maze[zero_idx[start_idx][0], zero_idx[start_idx][1]] = 2

    # Randomlty rotate the maze
    rot = np.random.randint(0, 4)
    maze = np.rot90(maze, rot)

    return maze