import numpy as np
import torch
import sys

# Print the current working directory
print("Current path:")
print(sys.path[0])

from mazes import *
import argparse
import os
import yaml
from goal_setters import random_goal
from maze_env import MutableMaze
from helpers import Net, ConvNet, generate_maze

eval_types = ["samples", "size"]


def generate_evaluation_set(config, seed=184):
    # Set the seed
    np.random.seed(seed)

    if "maze" in config["base"].keys():
        maze_name = config["base"]["maze"]
        if maze_name == "big":
            maze = big_maze
        elif maze_name == "huge":
            maze = huge_maze
        elif maze_name == "small":
            maze = small_maze
        else:
            raise ValueError("Invalid maze name.")
    else:
        if config["base"]["size"] is not None:
            maze_size = config["base"]["size"]
            print("Generating maze")
            maze = generate_maze(maze_size)
        else:
            maze = generate_maze(10)

    N = maze.shape[0]
    r = config["base"]["randomize"]
    num_eval_samples = config["eval"]["num_eval_samples"]

    data = np.zeros((num_eval_samples, N, N), dtype=int)

    if r == "g":
        for i in range(num_eval_samples):
            goal = random_goal(maze)
            maze[maze == 3] = 0
            maze[goal[0], goal[1]] = 3
            data[i] = np.copy(maze)
    elif r == "m":
        for i in range(num_eval_samples):
            data[i] = np.copy(generate_maze(N))

    return data


def split_by_samples(config, save_string):
    save_string_split = save_string.split("_")

    # Find the index of "samples"
    print(save_string)
    print(save_string_split)
    samples_idx = save_string_split.index("samples")
    num_expert_samples = int(save_string_split[samples_idx + 1])

    slash_split = save_string.split("/")
    real_save_string = slash_split[-1]

    model_path = (
        f"logs/{config['base']['save_dir']}/{real_save_string}/{real_save_string}.pt"
    )
    return num_expert_samples, model_path


def split_by_size(config, save_string):
    save_string_split = save_string.split("_")

    # Find the index of "samples"
    print(save_string)
    print(save_string_split)
    boardxboard = save_string_split[1]
    # Take boardxboard and split by x
    boardxboard_split = boardxboard.split("x")
    maze_size = int(boardxboard_split[0])

    slash_split = save_string.split("/")
    real_save_string = slash_split[-1]

    model_path = (
        f"logs/{config['base']['control']}/{real_save_string}/{real_save_string}.pt"
    )
    return maze_size, model_path


def evaluate(config, save_string, log_file, splitby="samples"):
    # Generate the evaluation set
    data = generate_evaluation_set(config)

    # Split along underscores
    save_string_split = save_string.split("_")

    # Find the index of "samples"
    print(save_string)
    print(save_string_split)

    if splitby == "samples":
        x_axis, model_path = split_by_samples(config, save_string)
    elif splitby == "size":
        x_axis, model_path = split_by_size(config, save_string)
    else:
        raise ValueError("Invalid split-by value.")

    # Load the model
    state_dict = torch.load(model_path)

    if config["base"]["network"] == "fc":
        model = Net(4)
    elif config["base"]["network"] == "cnn":
        model = ConvNet(config["base"]["size"])

    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    reward = 0

    for i in range(data.shape[0]):
        # Reset the environment
        env = MutableMaze(
            board_size=data[i][0].shape[0],
            init_grid_string=data[i],
            H=100,
            network=config["base"]["network"],
            render_mode="rgb_array",
        )
        obs = env.reset(data[i])

        # Run the model
        while True:
            # Get the action
            action = model(torch.Tensor(obs).unsqueeze(0)).argmax().item()

            # Take the action
            obs, r, term, trunc, info = env.step(action)

            # Render the environment
            env.render()

            # Update the reward
            reward += r

            if term or trunc:
                break

    # Compute the average reward
    reward /= data.shape[0]

    # Append the (num_expert_samples, reward) pair to the log file csv
    with open(log_file, "a") as f:
        print(f"Writing {x_axis}, {reward} to {log_file}")
        f.write(f"{x_axis}, {reward}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a set of arguments")
    parser.add_argument(
        "-y",
        "--config",
        type=str,
        required=True,
        help="Configuration file to run from. Any additional arguments will overwrite config.",
    )
    parser.add_argument(
        "-s",
        "--save_string",
        type=str,
        required=True,
        help="The string corresponding to your saved model.",
    )
    parser.add_argument(
        "-n",
        "--log_name",
        type=str,
        default=None,
        help="The name of the log file to save to.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="samples",
        choices=eval_types,
        help="The type of split to use for evaluation.",
    )

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = f"{args.save_string}_saved_eval.csv"

    # Check if the log file already exists
    if not os.path.exists(args.log_name):
        # If it doesn't, create it and write the header
        with open(args.log_name, "w") as f:
            f.write(f"{args.split}, reward\n")

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    evaluate(config, args.save_string, args.log_name, splitby=args.split)
