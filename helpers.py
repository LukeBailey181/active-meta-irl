import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.colors import ListedColormap

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


# Returns the transition matrix for a given maze
#     T[i * board_size + j][action][k * board_size + l] = P((k,l) | (i,j), action)
def get_transition_matrix(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N**2, 4, N**2))

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i, j] == 1 or grid_string[i, j] == 3:
                print(i * N + j)
                T[i * N + j][:, i * N + j] = 1
                continue

            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    print(i, j)
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

    return optimal_moves.T


# Visualizes the optimal moves for a given maze
def visualize_optimal_moves(maze, optimal_moves, save=False):
    # Plot the maze so that walls are black, empty spaces are white, the goal is green, and the start is blue
    cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])

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
