import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from maze_env import MutableMaze
import matplotlib.pyplot as plt
from goal_setters import random_goal
from mazes import *
import os
import yaml
from helpers import Net, ConvNet, ConvNetMC
from maze_env import Trajectory


def BehaviorCloning(
    train_dataset,
    test_dataset=None,
    env=None,
    config=yaml.load(open("configs/bc/bc_exp_cfg.yaml", "r"), Loader=yaml.FullLoader),
    N=10,
):
    r = config["base"]["randomize"]
    epochs = config["bc"]["num_epochs"]
    batch_size = config["bc"]["batch_size"]
    lr = config["bc"]["learning_rate"]
    save_weights = config["bc"]["save_weights"]
    eval_freq = config["bc"]["eval_freq"]
    num_eval_runs = config["bc"]["num_eval_runs"]
    network = config["base"]["network"]
    save_dir = config["base"]["save_dir"]

    if torch.cuda.is_available():
        print("------------------Training on GPU------------------")
        device = torch.device("cuda:0")
    else:
        print(
            "------------------No device available, training on CPU------------------"
        )
        device = torch.device("cpu")

    if save_dir is None:
        save_dir = "bc"

    if network == "fc":
        state_size = train_dataset[0][0].shape[0]
        net = Net(state_size)
    elif network == "cnn":
        state_size = train_dataset[0][0].shape[0]
        net = ConvNet(state_size)
    elif network == "mc_cnn":
        state_size = train_dataset[0][0].shape[1]
        net = ConvNetMC(state_size)

    net.to(device)

    # Turn the train dataset from an Nx|S|x4 array into a torch dataset
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_dataset[0]).float(), torch.tensor(train_dataset[1]).long()
    )
    if test_dataset is not None:
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_dataset[0]).float(), torch.tensor(test_dataset[1]).long()
        )

    # define info for logging
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    eval_reward = []
    logged_epochs = []

    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    # Train the network

    mod = eval_freq if eval_freq is not None else epochs // 10

    for epoch in range(epochs):
        if epoch % mod == 0:
            print(f"Epoch: {epoch} / {epochs}")
        # Training loop
        epoch_train_lossses = []
        epoch_train_accs = []
        for i, (state, action) in enumerate(train_loader):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                state = state.cuda()
                action = action.cuda()

            output = net(state)
            if state.shape[0] != 1:
                action = action.squeeze()

            loss = criterion(output, action)

            # If output isn't 2D, unsqueeze it
            if output.shape == (4,):
                output = output.unsqueeze(0)
                total = 1
            else:
                total = action.size(0)

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == action).sum().item()
            acc = correct / total
            epoch_train_lossses.append(loss.item())
            epoch_train_accs.append(acc)

            loss.backward()
            optimizer.step()

        train_losses.append(np.mean(epoch_train_lossses))
        train_accs.append(np.mean(epoch_train_accs))

        # Evaluation loop
        with torch.no_grad():
            if eval_freq is not None:
                if epoch % eval_freq == 0:
                    logged_epochs.append(epoch)
                    if test_dataset is None:
                        print("No test dataset provided, skipping evaluation")
                    else:
                        epoch_test_lossses = []
                        epoch_test_accs = []
                        for state, action in test_loader:
                            if torch.cuda.is_available():
                                state = state.cuda()
                                action = action.cuda()
                            output = net(state)
                            if state.shape[0] != 1:
                                action = action.squeeze()

                            loss = criterion(output, action)

                            # If output isn't 2D, unsqueeze it
                            if output.shape == (4,):
                                output = output.unsqueeze(0)
                                total = 1
                            else:
                                total = action.size(0)

                            _, predicted = torch.max(output.data, 1)
                            correct = (predicted == action).sum().item()
                            acc = correct / total

                            epoch_test_lossses.append(loss.item())
                            epoch_test_accs.append(acc)

                        test_losses.append(np.mean(epoch_test_lossses))
                        test_accs.append(np.mean(epoch_test_accs))

                    # Evaluate the policy
                    if eval_freq is not None:
                        batch_reward = []
                        obs = env.reset()

                        for i in range(num_eval_runs):
                            reward_sum = 0
                            while True:
                                if torch.cuda.is_available():
                                    obs = torch.tensor(obs).float().unsqueeze(0).cuda()
                                    action = net(obs).argmax().item()
                                else:
                                    action = (
                                        net(torch.tensor(obs).float().unsqueeze(0))
                                        .argmax()
                                        .item()
                                    )
                                obs, reward, term, trunc, info = env.step(int(action))
                                env.render()
                                reward_sum += reward
                                if term or trunc:
                                    break

                            if r == "g":
                                env.set_goal(random_goal(env))
                                obs = env.reset()
                            elif r == "m":
                                obs = env.reset(
                                    grid_string=generate_maze(env.board_size)
                                )
                            else:
                                obs = env.reset()

                            batch_reward.append(reward_sum)

                        eval_reward.append(np.mean(batch_reward))

    # Visualization
    plt.figure()

    # Label the whole figure
    plt.suptitle(
        f"{env.grid_string.shape[0]}x{env.grid_string.shape[0]} Grid, {N} Expert Samples"
    )
    # Create three subplots for train/test loss, train/test accuracy, and evaluation reward
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, "b-", label="Train Loss")
    if test_dataset is not None:
        plt.plot(logged_epochs, test_losses, "r-", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(train_accs, "b-", label="Train Accuracy")
    if test_dataset is not None:
        plt.plot(logged_epochs, test_accs, "r-", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(logged_epochs, eval_reward, "r-", label="Evaluation Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()

    save_string = f"bc_{env.grid_string.shape[0]}x{env.grid_string.shape[0]}_r_{r}_samples_{N}_batch_{batch_size}_epochs_{epochs}"

    # Make save_string directory
    if not os.path.exists(f"logs/{save_dir}/{save_string}"):
        os.makedirs(f"logs/{save_dir}/{save_string}")

    # Save the model
    if save_weights:
        torch.save(
            net.state_dict(), f"logs/{save_dir}/{save_string}/" + save_string + ".pt"
        )

    plt.savefig(f"logs/{save_dir}/{save_string}/" + save_string + ".png")

    # Save the data as a numpy array
    np.savez(
        f"logs/{save_dir}/{save_string}/" + save_string + ".npz",
        train_losses=train_losses,
        test_losses=test_losses,
        train_accs=train_accs,
        test_accs=test_accs,
        eval_reward=eval_reward,
        logged_epochs=logged_epochs,
    )

    # plt.show()


def generateExpertDataset(env, r="", num_train_samples=50, num_test_samples=10):
    train_trajectories = []
    test_trajectories = []
    for i in range(num_train_samples):
        train_trajectories.append(generateExpertTrajectory(env, r=r))

    if num_test_samples is not None:
        for i in range(num_test_samples):
            test_trajectories.append(generateExpertTrajectory(env, r=r))

    train_states = []
    train_actions = []
    for trajectory in train_trajectories:
        for state, action, _ in trajectory.transitions():
            train_states.append(state)
            train_actions.append(action)

    test_states = []
    test_actions = []
    for trajectory in test_trajectories:
        for state, action, _ in trajectory.transitions():
            test_states.append(state)
            test_actions.append(action)

    # Convert to numpy arrays
    train_states = np.array(train_states)
    train_actions = np.array(train_actions)
    test_states = np.array(test_states)
    test_actions = np.array(test_actions)

    train_dataset = (train_states, train_actions)
    test_dataset = (test_states, test_actions)

    if num_test_samples is None:
        test_dataset = None

    return train_dataset, test_dataset


def generateExpertTrajectory(env, r="", maze=None):
    if maze is not None:
        obs = env.reset(grid_string=maze)
        obs_s = env.get_state_obs()
    elif r == "g":
        env.set_goal(random_goal(env))
        obs = env.reset()
        obs_s = env.get_state_obs()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
        obs_s = env.get_state_obs()
    else:
        obs = env.reset()
        obs_s = env.get_state_obs()

    policy = solve_maze(env.grid_string)

    cur_trajectory = Trajectory()
    while True:
        action = policy[obs_s[0], obs_s[1]]

        obs_old = obs

        obs, reward, term, trunc, info = env.step(action)
        obs_s = env.get_state_obs()

        cur_trajectory.add_transition(list(obs_old), action, obs)

        env.render()

        if term or trunc:
            break

    return cur_trajectory
