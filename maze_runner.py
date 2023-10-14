from maze_env import MutableMaze
import numpy as np
from manual_controller import ManualControl
from goal_setters import random_goal
from mazes import *
from helpers import *
import argparse
from algos.bc import BehaviorCloning, generateExpertDataset
import yaml

control_options = ["manual", "random", "policy", "expert", "bc"]
randomization_options = ["g", "m", ""]
maze_options = ["small", "big", "huge"]


# Control the agent using keyboard input
def manualController(env, config):
    manual_control = ManualControl(env, seed=42, set_goal=True)
    manual_control.start()


def bcController(env, config):
    if config["base"]["num_expert_samples"] is not None:
        num_train_samples = config["base"]["num_expert_samples"]
    elif config["base"]["num_train_samples"] is not None:
        num_train_samples = config["base"]["num_train_samples"]
    else:
        num_train_samples = 500

    num_test_samples = config["bc"]["num_test_samples"]

    r = config["base"]["randomize"]

    train_dataset, test_dataset = generateExpertDataset(
        env, num_train_samples=num_train_samples, num_test_samples=num_test_samples, r=r
    )

    BehaviorCloning(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        env=env,
        config=config,
    )


# Control the agent using random actions
def randomController(env, config):
    while True:
        # action = env.action_space.sample()
        action = np.random.choice([0, 1, 2, 3])
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            env.set_goal(random_goal(env))
            env.reset()


# Control the agent using the optimal policy
def expertController(env, config):
    r = config["base"]["randomize"]
    _, _ = generateExpertDataset(
        env, num_train_samples=100, r=r
    )


# Control the agent using a reinforcement learning algorithm
def rlController(env, config):
    print("Not implemented")


def main(config, maze_init):
    # Define maze environment

    render_mode = "rgb_array" if config["base"]["headless"] else "human"

    env = MutableMaze(
        board_size=maze_init.shape[0],
        init_grid_string=maze_init,
        H=100,
        render_mode=render_mode,
    )

    if config["base"]["control"] == "manual":
        manualController(env, config)
    elif config["base"]["control"] == "random":
        randomController(env, config)
    elif config["base"]["control"] == "policy":
        rlController(env, config)
    elif config["base"]["control"] == "expert":
        expertController(env, config)
    elif config["base"]["control"] == "bc":
        bcController(env, config)
    else:
        print("Invalid control type")


def update_config(config, args):
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)

    # For each arg key
    keys = vars(args).keys()
    for key in keys:
        # If the arg is not None, update the config
        if vars(args)[key] is not None:
            config["base"][key] = vars(args)[key]
    
    # If any of the options aren't in args, add default values to config
    if config["base"]["control"] is None:
        config["base"]["control"] = "expert"
        config["base"]["num_expert_samples"] = 100
    if config["base"]["randomize"] is None:
        config["base"]["randomize"] = ""
    if config["base"]["size"] is None:
        config["base"]["size"] = None
    if config["base"]["maze"] is None and args.size is None:
        config["base"]["maze"] = "big"

    return config

def create_config(args):
    keys = vars(args).keys()
    config = {}
    config["base"] = {}
    config["base"]["control"] = args.control
    config["base"]["num_expert_samples"] = None
    config["base"]["randomize"] = args.randomize
    config["base"]["size"] = args.size
    config["base"]["maze"] = args.maze
    config["base"]["config"] = args.config
    config["base"]["headless"] = False
    config["base"]["H"] = 100

    # If any of the options aren't in args, add default values to config
    if args.control is None:
        config["base"]["control"] = "expert"
        config["base"]["num_expert_samples"] = 100
    if args.randomize is None:
        config["base"]["randomize"] = ""
    if args.size is None:
        config["base"]["size"] = None
    if args.maze is None and args.size is None:
        config["base"]["maze"] = "big"
    if args.config is None:
        config["base"]["config"] = None
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a maze controller")
    parser.add_argument(
        "-c",
        "--control",
        type=str,
        choices=control_options,
        help="Control type: manual, random, policy, or expert",
    )
    parser.add_argument(
        "-r",
        "--randomize",
        type=str,
        choices=randomization_options,
        help='Randomize the maze. Specify "g" for goal, "m" for maze, and "" for none.',
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        help="Size of the maze. Defaults to current maze size if fixed maze is specified.",
    )
    parser.add_argument(
        "-m",
        "--maze",
        type=str,
        choices=maze_options,
        help="Fixed maze to use. Limited to mazes found in mazes.py.",
    )

    parser.add_argument(
        "-N",
        "--num_expert_samples",
        type=int,
        help="Number of expert samples to draw for IL/IRL. Does not include test samples.",
    )

    parser.add_argument(
        "-y",
        "--config",
        type=str,
        default=None,
        help="Configuration file to run from. Any additional arguments will overwrite config.",
    )

    args = parser.parse_args()
    if args.config is None:
        config = create_config(args)
    else:
        config = update_config(args.config, args)

    print("Configuration: ")
    print(config)

    # Ensure that the arguments are valid
    if config["base"]["randomize"] == "g" and not config["base"]["maze"] and not config["base"]["size"]:
        print("Cannot randomize goal without specifying a fixed maze or size.")
        exit()
    elif config["base"]["randomize"] == "g" and config["base"]["maze"]:
        if config["base"]["maze"] == "small":
            maze = small_maze
        elif config["base"]["maze"] == "big":
            maze = big_maze
        else:
            maze = huge_maze
        if config["base"]["size"]:
            print("WARNING: Ignoring arg size since a fixed maze was specified.")
    elif config["base"]["randomize"] == "g" and config["base"]["size"]:
        maze = generate_maze(config["base"]["size"])
    elif config["base"]["randomize"] == "m" and not config["base"]["size"]:
        print("Cannot randomize maze without specifying a size.")
        exit()
    elif config["base"]["randomize"] == "m" and config["base"]["size"]:
        maze = generate_maze(config["base"]["size"])
        if config["base"]["maze"]:
            print("WARNING: Ignoring arg maze since maze randomization was specified.")
    elif config["base"]["maze"]:
        if config["base"]["maze"] == "small":
            maze = small_maze
        elif config["base"]["maze"] == "big":
            maze = big_maze
        else:
            maze = huge_maze
        if config["base"]["size"]:
            print("WARNING: Ignoring arg size since a fixed maze was specified.")
    elif config["base"]["size"]:
        maze = generate_maze(config["base"]["size"])
    else:
        print("ERROR: No maze or size specified.")
        exit()

    main(config, maze)
