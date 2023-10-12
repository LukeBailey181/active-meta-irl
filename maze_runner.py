from maze_env import MutableMaze
import numpy as np
from minigrid.manual_control import ManualControl

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

env = MutableMaze(
    board_size=10,
    init_grid_string=maze,
    H=200,
    render_mode='human',)

# for i in range(200):
#     action = env.action_space.sample()
#     obs, reward, term, trunc, info = env.step(action)
#     print(obs)
#     env.render()
#     if term or trunc:
#         env.reset()
        

# enable manual control for testing
manual_control = ManualControl(env, seed=42)
manual_control.start()