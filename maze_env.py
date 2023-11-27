from __future__ import annotations

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from itertools import chain
from helpers import get_transition_matrix, tuple_to_key, key_to_action
import time
from tqdm import tqdm


# A maze class which allows the user to specify the maze layout
class MutableMaze(MiniGridEnv):
    def __init__(
        self,
        board_size=10,
        init_grid_string=None,
        H=None,
        **kwargs,
    ):
        # Used to track agent position in the superclass
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        self.step_count = 0

        # Size of our board (NxN)
        self.board_size = board_size

        # The maze is specified as a 2D integer array
        init_grid_string = np.array(init_grid_string).reshape((board_size, board_size))
        self.grid_string = init_grid_string.T

        self.goal_pos = np.argwhere(self.grid_string == 3)[0]

        # IDK what this does --- for the superclass
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Time horizon
        if H is None:
            H = 4 * board_size**2
        self.H = H

        # Transition
        #self.T_matrix = get_transition_matrix(self.grid_string)
        #print("Forming transition")

        self.T_matrix = self.get_t_matrix()
        #print("Transition formed")

        # Initialize the superclass
        super().__init__(
            mission_space=mission_space,
            grid_size=board_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=H,
            **kwargs,
        )

        # Reset the board
        self.reset()

    # IDK --- used by the superclass
    @staticmethod
    def _gen_mission():
        return "grand mission"

    # Check if pos is inside the box with upper-left corner ul and bottom-right corner br
    def _in_box(self, pos, ul, br):
        return (
            pos[0] >= ul[0] and pos[0] <= br[0] and pos[1] >= ul[1] and pos[1] <= br[1]
        )

    # Set the grid string
    def set_grid_string(self, string):
        string = np.array(string)

        # Check that each character is in {0,1,2,3}, and 2 and 3 appear exactly once
        if not (
            set(list(string.flatten())) <= set([0, 1, 2, 3])
            and np.sum(string == 2) == 1
            and np.sum(string == 3) == 1
        ):
            print("Invalid grid string. Reverting to default.")
            return False

        # Check that the length matches the grid dimensions
        if string.shape[0] * string.shape[1] == self.board_size**2:
            # Shape the string into a square
            string = string.reshape((self.board_size, self.board_size)).T
            self.grid_string = string

            return True
        else:
            print(len(string), self.board_size**2)
            print("Invalid grid size. Reverting to default.")
            return False

    # Set the goal position
    def set_goal(self, goal_pos):
        x, y = goal_pos

        if not self._in_box(
            goal_pos, (1, 1), (self.board_size - 1, self.board_size - 1)
        ):
            print("Position outside of board. Reverting to default.")
            return False

        if self.grid_string[x][y] == 0:
            current_goal_pos = np.argwhere(self.grid_string == 3)[0]
            self.grid_string[current_goal_pos[0]][current_goal_pos[1]] = 0
            self.grid_string[x][y] = 3
            return True

        # Otherwise
        print("Position occupied. Reverting to default.")
        return False

    # Set the maze using the grid string
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

    # Reset the environment
    def reset(
        self,
        grid_string=None,
        **kwargs,
    ):
        if grid_string is not None:
            self.set_grid_string(grid_string)
            self.T_matrix = get_transition_matrix(self.grid_string)

        super().reset()

        self.goal_pos = np.argwhere(self.grid_string == 3)[0]

        return [
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
        ]

    # Step the environment
    def step(self, action):
        self.agent_dir = action
        # print(action)
        obs, reward, term, trunc, info = super().step(2)
        self.step_count += 1

        reward = int(reward != 0)

        obs_refined = [
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
        ]

        return obs_refined, reward, term, trunc, info

    def get_num_states(self):
        """Return number of states"""

        # For now fixing this to (x, y, x_goal, y_goal)
        return self.board_size ** 4

    def get_num_actions(self):
        return 4


    def get_features(self): 
        """Return matrix of feature values"""

        # For now using simply 1 hot encoding for the features, 
        # mayby later this can be some clever embedding.
        return np.identity(self.get_num_states())
    
    def state_to_index(self, state): 

        # Convert state into index of state. 
        # For now states are (x,y, x_goal, y_goal)
        x, y, x_goal, y_goal = state
        N = self.board_size

        # Convert to base n
        return (x * N**3) + (y * N**2) + (x_goal * N) + y_goal

    def index_to_state(self, index):

        M = index

        N = self.board_size
        x = M // N**3
        M_prime = M % N**3
        y = M_prime // N**2
        M_double_prime = M_prime % N**2
        x_goal = M_double_prime // N
        y_goal = M_double_prime % N
        
        return (x, y, x_goal, y_goal)

    def T_func(self, state_from, action, state_to): 

        x_from,y_from,_,_ = state_from
        x_to,y_to,_,_ = state_to

        grid_string = self.grid_string.T
        from_void = grid_string[x_from, y_from] == 0 or grid_string[x_from, y_from] == 2
        to_void = grid_string[x_to, y_to] == 0 or grid_string[x_to, y_to] == 2
        to_goal = grid_string[x_to, y_to] == 3

        change_to_action = {
            (0, -1): 3,
            (0, 1): 1,
            (1, 0): 0,
            (-1, 0): 2
        }

        change = (x_to - x_from, y_to - y_from)
        if change not in change_to_action: 
            # Not a move by 1 square
            return 0

        if change_to_action[change] == action:
            # Might be valid
            if (from_void and to_void) or (from_void and to_goal): 
                # Can do this if moving 1 square and action correct
                return 1
        return 0

    def T_func_index(self, index_from, action, index_to): 

        state_from = self.index_to_state(index_from)
        state_to = self.index_to_state(index_to)

        return self.T_func(state_from, action, state_to)
    
    def get_t_matrix(self): 

        # Get the transition matrix for a given action
        # This is a matrix of size (n_states, n_states)
        # where entry (i,j) is the probability of transitioning from state i to state j
        # when taking action a. 

        n_states = self.get_num_states()
        n_actions = self.get_num_actions()
        T = np.zeros((n_states, n_actions, n_states))

        action_to_change = {
            3: (0, -1),
            1: (0, 1),
            0: (1, 0),
            2: (-1, 0)
        }

        grid_string = self.grid_string.T
        print("Constructing T matrix")
        for state_from in tqdm(range(n_states)): 
            (x_from,y_from,x_goal,y_goal) = self.index_to_state(state_from)

            from_void_or_start = grid_string[x_from, y_from] == 0 or grid_string[x_from, y_from] == 2

            # For each action this should be distribution over next states 
            for action in range(n_actions): 
                # If the block is wall, or goal, then any action keeps you here 
                if not from_void_or_start:
                    T[state_from, action, state_from] = 1
                else: 
                    # Get the valid change, if it exists
                    x_dif, y_dif = action_to_change[action]
                    x_to = x_from + x_dif
                    y_to = y_from + y_dif
                    # Check if this is a valid state, else you will state in same state 
                    to_type = grid_string[x_to, y_to]
                    if x_to < 0 or x_to >= self.board_size or y_to < 0 or y_to >= self.board_size:
                        # Outside of grid, stay in same state
                        T[state_from, action, state_from] = 1
                    elif to_type == 1:
                        # Cannot move to wall 
                        T[state_from, action, state_from] = 1
                    else: 
                        # Moving to void, start, or goal
                        state_to = self.state_to_index((x_to, y_to, x_goal, y_goal))
                        T[state_from, action, state_to] = 1

        # Test transition is valid
        for state_from in range(n_states): 
            for action in range(n_actions): 
                assert np.sum(T[state_from, action, :]) == 1

        return T

        #print("Constructing T matrix")
        #for state_from in tqdm(range(n_states)): 

        #    # Loop over neighboring states
        #    (x,y,x_goal,y_goal) = self.index_to_state(state_from)

        #    # Neighbors are (x+1,y), (x-1,y), (x,y+1), (x,y-1)
        #    for change in [(1,0), (-1,0), (0,1), (0,-1)]:
        #        x_to = x + change[0]
        #        y_to = y + change[1]

        #        # Check if this is a valid state
        #        if x_to < 0 or x_to >= self.board_size or y_to < 0 or y_to >= self.board_size:
        #            continue

        #        state_to = self.state_to_index((x_to, y_to, x_goal, y_goal))

        #        # Loop over actions
        #        for action in range(n_actions): 
        #            tmp = self.T_func_index(state_from, action, state_to)
        #            if tmp is None: 
        #                breakpoint()
        #            T[state_from, action, state_to] = tmp 

        return T

        


class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.

    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """
    def __init__(self, transitions=None):
        if transitions is None:
            self._t = []
        else: 
            self._t = transitions

    def add_transition(self, state_from, action, state_to): 
        self._t.append((state_from, action, state_to))

    def transitions(self):
        """
        The transitions of this trajectory.

        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return self._t

    def states(self):
        """
        The states visited in this trajectory.

        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)