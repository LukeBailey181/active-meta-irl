from maze_env import MutableMaze
import numpy as np
from manual_controller import ManualControl
from goal_setters import random_goal
from mazes import *
from helpers import *
import argparse

control_options = ['manual', 'random', 'policy', 'expert']
randomization_options = ['g', 'm', '']
maze_options = ['small', 'big', 'huge']


# Control the agent using keyboard input
def manualController(env, r=""):
    manual_control = ManualControl(env, seed=42, set_goal=True)
    manual_control.start()


# Control the agent using random actions
def randomController(env, r=""):
    while True:
        # action = env.action_space.sample()
        action = np.random.choice([0,1,2,3])
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            env.set_goal(random_goal(env))
            env.reset()


# Control the agent using the optimal policy
def expertController(env, r="", vis=False):
    # Randomize either the goal or the maze
    if r == 'g':
        env.set_goal(random_goal(env))
        env.reset()
    elif r == 'm':
        env.reset(grid_string=generate_maze(env.board_size))

    # NxN matrix of optimal actions
    policy = solve_maze(env.grid_string)
    
    # Start position
    init = np.argwhere(big_maze == 2)[0]

    # First step
    action = policy[init[0], init[1]]
    obs, reward, term, trunc, info = env.step(action)
    env.render()

    if vis:
        visualize_optimal_moves(env.grid_string, policy, save=True)

    while True:
        action = policy[obs[0], obs[1]]
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            if r == 'g':
                env.set_goal(random_goal(env))
                env.reset()
            elif r == 'm':
                env.reset(grid_string=generate_maze(env.board_size))

            policy = solve_maze(env.grid_string)

            # Check that the policy is valid
            try:
                assert np.all(policy != -1)
            except:
                print("INVALID POLICY !!!!! THIS SHOULD NEVER HAPPEN")
                print(policy)
                print(env.grid_string)
                break

            if vis:
                visualize_optimal_moves(env.grid_string, policy)
            
            obs = np.argwhere(env.grid_string == 2)[0]


# Control the agent using a reinforcement learning algorithm
def rlController(env, r=""):
    print("Not implemented")


def main(args, maze_init):
    # Define maze environment
    env = MutableMaze(
        board_size=maze_init.shape[0],
        init_grid_string=maze_init,
        H=200,
        render_mode='human',)

    vis = True

    if args.control == 'manual':
        manualController(env, r=args.randomize)
    elif args.control == 'random':
        randomController(env, r=args.randomize)
    elif args.control == 'policy':
        expertController(env, r=args.randomize)
    elif args.control == 'expert':
        expertController(env, r=args.randomize, vis=vis)
    else:
        print("Invalid control type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a maze controller")
    parser.add_argument(
        "-c", 
        "--control", 
        type=str,
        default="expert",
        choices=control_options,
        help="Control type: manual, random, policy, or expert")
    parser.add_argument(
        '-r',
        '--randomize',
        type=str,
        default='',
        choices=randomization_options,
        help='Randomize the maze. Specify "g" for goal, "m" for maze, and "" for none.')
    parser.add_argument(
        '-s',
        '--size',
        type=int,
        default=None,
        help='Size of the maze. Defaults to current maze size if fixed maze is specified.')
    parser.add_argument(
        '-m',
        '--maze',
        type=str,
        default=None,
        choices=maze_options,
        help='Fixed maze to use. Limited to mazes found in mazes.py.'
    )
    
    args = parser.parse_args()
    # Ensure that the arguments are valid
    if args.randomize == 'g' and not args.maze and not args.size:
        print("Cannot randomize goal without specifying a fixed maze or size.")
        exit()
    elif args.randomize == 'g' and args.maze:
        if args.maze == 'small':
            maze = small_maze
        elif args.maze == 'big':
            maze = big_maze
        else:
            maze = huge_maze
        if args.size:
            print("WARNING: Ignoring arg size since a fixed maze was specified.")
    elif args.randomize == 'g' and args.size:
        maze = generate_maze(args.size)
    elif args.randomize == 'm' and not args.size:
        print("Cannot randomize maze without specifying a size.")
        exit()
    elif args.randomize == 'm' and args.size:
        maze = generate_maze(args.size)
        if args.maze:
            print("WARNING: Ignoring arg maze since maze randomization was specified.")
    elif args.maze: 
        if args.maze == 'small':
            maze = small_maze
        elif args.maze == 'big':
            maze = big_maze
        else:
            maze = huge_maze
        if args.size:
            print("WARNING: Ignoring arg size since a fixed maze was specified.")
    elif args.size:
        maze = generate_maze(args.size)
    else:
        print("ERROR: No maze or size specified.")
        exit()

    main(args, maze)