from maze_env import MutableMaze
import numpy as np
# from minigrid.manual_control import ManualControl
from manual_controller import ManualControl
from goal_setters import random_goal

# Between 'manual' and 'random'
mode = 'manual'

# Fixed maze
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

if mode == 'random':
    for i in range(200):
        # action = env.action_space.sample()
        action = np.random.choice([0,1,2,3])
        obs, reward, term, trunc, info = env.step(action)
        print(obs)
        env.render()
        if term or trunc:
            env.set_goal(random_goal(env))
            env.reset()
elif mode == 'manual':
    manual_control = ManualControl(env, seed=42, set_goal=True)
    manual_control.start()
else:
    print("Not implemented")