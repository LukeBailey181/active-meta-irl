from maze_env import MutableMaze
import numpy as np
# from minigrid.manual_control import ManualControl
from manual_controller import ManualControl


maze = np.array([
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

def select_goal(env):
    # Grid_string is a 2D array of integers
    grid_string = env.grid_string
    # find all the 0s in the grid_string
    zero_positions = (grid_string == 0).T

    num_zeros = np.sum(zero_positions)

    zero_idx = np.random.choice(num_zeros)

    # Find the 2D coordinates of the zero_idx'th zero
    zero_coords = np.argwhere(zero_positions)

    goal = zero_coords[zero_idx]

    # Choose a random 2D position with a 0


    return goal

env = MutableMaze(
    board_size=10,
    init_grid_string=maze,
    H=200,
    render_mode='human',)

# for i in range(10):
#     print(select_goal(env))

# for i in range(200):
#     # action = env.action_space.sample()
#     action = np.random.choice([0,1,2,3])
#     obs, reward, term, trunc, info = env.step(action)
#     print(obs)
#     env.render()
#     if term or trunc:
#         env.set_goal(select_goal(env))
#         env.reset()

# for i in range(200):
#     # action = env.action_space.sample()
#     action = np.random.choice([0,1,2,3])
#     obs, reward, term, trunc, info = env.step(action)
#     print(obs)
#     env.render()
#     if term or trunc:
#         env.set_goal(select_goal(env))
#         env.reset()
        
        

# enable manual control for testing
manual_control = ManualControl(env, seed=42, set_goal=True)
manual_control.start()