import numpy as np


# Randomly select an open location
def random_goal(env):
    # Grid_string is a 2D array of integers
    if type(env) == np.ndarray:
        grid_string = env
    else:
        grid_string = env.grid_string
    # find all the 0s in the grid_string
    zero_positions = np.argwhere(grid_string == 0)

    # Randomly choose one of them
    zero_idx = np.random.choice(zero_positions.shape[0])
    goal = zero_positions[zero_idx]

    return goal
