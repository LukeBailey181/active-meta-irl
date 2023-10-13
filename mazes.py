import numpy as np
from helpers import solve_maze

#################################################################################
###  This file contains a set of mazes and functions to generate novel mazes  ###
#################################################################################

def generate_maze(n):
    # Generate a (possibly impossible) maze
    def _gen_try(n):
        maze = np.ones((n, n), dtype=int)
        maze[1:-1, 1:-1] = 0
        for i in range(2, n-2, 2):
            for j in range(2, n-2, 2):
                maze[i, j] = 1
                if i == 2:
                    maze[i-1, j] = 1
                if j == n-3:
                    maze[i, j+1] = 3
                if np.random.randint(0, 2) == 0:
                    maze[i+1, j] = 1
                else:
                    maze[i, j+1] = 1
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

huge_maze = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Fixed 10 x 10 maze
big_maze = np.array([
  [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
  [ 1, 0, 1, 0, 0, 0, 0, 1, 3, 1 ],
  [ 1, 0, 1, 1, 1, 1, 0, 1, 0, 1 ],
  [ 1, 0, 1, 0, 1, 1, 0, 1, 0, 1 ],
  [ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
  [ 1, 1, 1, 0, 1, 1, 1, 0, 1, 1 ],
  [ 1, 0, 1, 0, 0, 1, 1, 0, 0, 1 ],
  [ 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 ],
  [ 1, 2, 0, 0, 0, 0, 0, 0, 0, 1 ],
  [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
])

small_maze = np.array([
  [ 1, 1, 1, 1, 1 ],
  [ 1, 0, 1, 3, 1 ],
  [ 1, 0, 1, 0, 1 ],
  [ 1, 0, 2, 0, 1 ],
  [ 1, 1, 1, 1, 1 ]
])

# Fixed 4 x 4 maze
tiny_maze = np.array([ 
    [1, 1, 1, 1],
    [1, 2, 3, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
])