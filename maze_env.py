from __future__ import annotations

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# A maze class which allows the user to specify the maze layout
class MutableMaze(MiniGridEnv):
    def __init__(
        self,
        board_size=10,
        init_grid_string=None,
        H = None,
        **kwargs,
    ):
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 0

        self.step_count = 0

        self.board_size = board_size

        init_grid_string = np.array(init_grid_string).reshape((board_size, board_size))
        self.grid_string = init_grid_string.T

        mission_space = MissionSpace(mission_func=self._gen_mission)


        if H is None:
            H = 4 * board_size**2

        self.H = H

        super().__init__(
            mission_space=mission_space,
            grid_size=board_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=H,
            **kwargs,
        )

        self.reset()

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # Check if pos is inside the box with upper-left corner ul and bottom-right corner br
    def _in_box(self, pos, ul, br):
        return (pos[0] >= ul[0] and pos[0] < br[0] and pos[1] >= ul[1] and pos[1] < br[1])

    def set_grid_string(self, string):
        string = np.array(string)

        # Check that each character is in {0,1,2,3}, and 2 and 3 appear exactly once
        if not (set(string) <= set([0, 1, 2, 3]) and np.sum(string == 2) == 1 and np.sum(string == 3) == 1):
            print("Invalid grid string. Reverting to default.")
            return False

        # Check that the length matches the grid dimensions
        if len(string) == self.board_size ** 2:
            # Shape the string into a square
            string = string.reshape((self.board_size, self.board_size)).T
            self.grid_string = string

            return True
        else:
            print("Invalid grid size. Reverting to default.")
            return False

    def set_goal(self, goal_pos):
        y, x = goal_pos

        if not self._in_box(goal_pos, (1, 1), (self.board_size - 1, self.board_size - 1)):
            print("Position outside of board. Reverting to default.")
            return False

        if self.grid_string[x][y] == 0:
            self.grid_string[x][y] = 3
            return True

        # Otherwise
        print("Position occupied. Reverting to default.")
        return False
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(self.board_size, self.board_size)

        for x in range(0, self.board_size):
            for y in range(0, self.board_size):
                cell_type = self.grid_string[x][y]
                if cell_type == 1:
                    self.grid.set(x, y, Wall())
                elif cell_type == 2:
                    self.agent_start_pos = (x, y)
                    self.agent_start_dir = 0
                elif cell_type == 3:
                    self.put_obj(Goal(), x, y)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    
    def step(self, action):
        # 0,1,2,3 = left, right, up, down

        self.agent_dir = action
        # print(action)
        obs, reward, term, trunc, info = super().step(2)
        self.step_count += 1

        obs_refined = [self.agent_pos[0], self.agent_pos[1], self.agent_dir]


        return obs_refined, reward, term, trunc, info

