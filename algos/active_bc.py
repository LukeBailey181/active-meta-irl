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
from helpers import Net
from maze_env import Trajectory


def heuristic(m1, m2):
    agent_1 = np.argwhere(m1 == 2)[0]
    agent_2 = np.argwhere(m2 == 2)[0]

    goal_1 = np.argwhere(m1 == 3)[0]
    goal_2 = np.argwhere(m2 == 3)[0]

    m1_blocks = m1 == 1
    m2_blocks = m2 == 1

    weights = [1, 1, 1]

    N = m1_blocks.shape[0]

    # Compute the manhattan distance between the two agents
    agent_manhattan = (
        np.abs(agent_1[0] - agent_2[0]) + np.abs(agent_1[1] - agent_2[1])
    ) / (2 * N)

    # Compute the manhattan distance between the two goals
    goal_manhattan = (np.abs(goal_1[0] - goal_2[0]) + np.abs(goal_1[1] - goal_2[1])) / (
        2 * N
    )

    overlap = np.sum(m1_blocks != m2_blocks) / N**2

    heuristic = (
        weights[0] * agent_manhattan
        + weights[1] * goal_manhattan
        + weights[2] * overlap
    )
    # Turn heuristic from a numpy array into a float

    return heuristic


def active_bc(env=None, budget=100, config=None):
    mazes = []

    env.set_goal(random_goal(env))
    init_maze = env.grid_string.T

    mazes.append(init_maze)

    init_traj = generateExpertTrajectory(
        env, r=config["base"]["randomize"], maze=init_maze
    )

    trajectories = [init_traj]

    # Generate the initial dataset
    state_action_tuples = [traj.transitions() for traj in trajectories]

    states = []
    actions = []

    for traj in state_action_tuples:
        for state, action, _ in traj:
            states.append(state)
            actions.append(action)

    np_states = np.array(states)
    np_actions = np.array(actions)

    print("States:")
    print(np_states)
    print("Actions:")
    print(np_actions)

    idx, net, failed_configs = BehaviorCloning(
        train_dataset=(np_states, np_actions),
        test_dataset=None,
        env=env,
        config=config,
        net=None,
        idx=0,
    )

    while len(trajectories) <= budget:
        print(f"Training on {len(trajectories)} expert samples")
        if len(failed_configs) != 0:
            max_heuristic = 0
            max_heuristic_idx = 0
            for t in range(len(failed_configs)):
                cur_maze_heuristics = np.zeros(len(mazes))
                maze_count = 0
                for maze in mazes:
                    cur_maze_heuristics[maze_count] = heuristic(maze, failed_configs[t])
                    maze_count += 1
                min_heuristic = np.min(cur_maze_heuristics)

                if min_heuristic > max_heuristic:
                    max_heuristic = min_heuristic
                    max_heuristic_idx = t

            print(f"Max heuristic: {max_heuristic}")
            cur_maze = failed_configs[max_heuristic_idx]
        else:
            env.set_goal(random_goal(env))
            cur_maze = env.grid_string.T

        cur_traj = generateExpertTrajectory(
            env, r=config["base"]["randomize"], maze=cur_maze
        )

        trajectories.append(cur_traj)
        state_action_tuples.append(cur_traj.transitions())

        for state, action, _ in cur_traj.transitions():
            states.append(state)
            actions.append(action)

        np_states = np.array(states)
        np_actions = np.array(actions)

        idx, net, failed_configs = BehaviorCloning(
            train_dataset=(np_states, np_actions),
            test_dataset=None,
            env=env,
            config=config,
            net=net,
            idx=(idx + 1),
        )

    # Save the final model

    # If logs/bc_al/al_model_{budget} doesn't exist, create it
    if not os.path.exists(f"logs/bc-al/bc_al_model_samples_{budget}"):
        os.makedirs(f"logs/bc-al/bc_al_model_samples_{budget}")

    torch.save(
        net.state_dict(),
        f"logs/bc-al/bc_al_model_samples_{budget}/bc_al_model_samples_{budget}.pt",
    )


def BehaviorCloning(
    train_dataset,
    test_dataset=None,
    env=None,
    config=yaml.load(open("configs/bc/bc_exp_cfg.yaml", "r"), Loader=yaml.FullLoader),
    net=None,
    idx=0,
):
    r = config["base"]["randomize"]
    epochs = config["bc"]["num_epochs"]
    batch_size = config["bc"]["batch_size"]
    lr = config["bc"]["learning_rate"]
    save_weights = config["bc"]["save_weights"]
    eval_freq = config["bc"]["eval_freq"]
    num_eval_runs = config["bc"]["num_eval_runs"]

    state_size = train_dataset[0][0].shape[0]

    eps = 1e-8

    if net is None:
        net = Net(state_size)

    # Turn the train dataset from an Nx|S|x4 array into a torch dataset
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_dataset[0]).float(), torch.tensor(train_dataset[1]).long()
    )

    # Print the size of the train dataset
    print(f"Train dataset size: {train_dataset.__len__()}")

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

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

    # Train the network

    mod = eval_freq if eval_freq is not None else epochs // 10
    mod = 500

    for epoch in range(epochs):
        if epoch % mod == 0:
            print(f"Epoch: {epoch} / {epochs}")
        # Training loop
        epoch_train_losses = []
        epoch_train_accs = []
        for i, (state, action) in enumerate(train_loader):
            optimizer.zero_grad()
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
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(acc)

            loss.backward()
            optimizer.step()

        train_losses.append(np.mean(epoch_train_losses))
        train_accs.append(np.mean(epoch_train_accs))

        if np.mean(epoch_train_losses) < eps:
            print("Loss is 0, stopping training")
            break

        # Evaluation loop
        with torch.no_grad():
            if False:
                if epoch % eval_freq == 0:
                    logged_epochs.append(epoch)
                    if test_dataset is None:
                        print("No test dataset provided, skipping evaluation")
                    else:
                        epoch_test_lossses = []
                        epoch_test_accs = []
                        for state, action in test_loader:
                            output = net(state)
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
                                action = net(torch.tensor(obs).float()).argmax().item()
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
        f"{env.grid_string.shape[0]}x{env.grid_string.shape[0]} Grid, {train_dataset.__len__()} Expert Samples"
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

    save_string = f"bc_{env.grid_string.shape[0]}x{env.grid_string.shape[0]}_r_{r}_samples_{train_dataset.__len__()}_batch_{batch_size}_epochs_{epochs}"

    # Make save_string directory
    if not os.path.exists(f"logs/bc/{save_string}"):
        os.makedirs(f"logs/bc/{save_string}")

    plt.savefig(f"logs/bc/{save_string}/" + save_string + f"_{idx}.png")

    # Save the data as a numpy array
    # np.savez(
    #     f"logs/bc/{save_string}/" + save_string + f"_{idx}.npz",
    #     train_losses=train_losses,
    #     test_losses=test_losses,
    #     train_accs=train_accs,
    #     test_accs=test_accs,
    #     eval_reward=eval_reward,
    #     logged_epochs=logged_epochs,
    # )

    failed_configs = []

    obs = env.reset()

    if r == "g":
        env.set_goal(random_goal(env))
        obs = env.reset()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
    else:
        obs = env.reset()

    for i in range(num_eval_runs):
        reward_sum = 0
        while True:
            action = net(torch.tensor(obs).float()).argmax().item()
            obs, reward, term, trunc, info = env.step(int(action))
            env.render()
            if term or trunc:
                if reward == 0:
                    failed_configs.append(env.grid_string)
                break

        if r == "g":
            env.set_goal(random_goal(env))
            obs = env.reset()
        elif r == "m":
            obs = env.reset(grid_string=generate_maze(env.board_size))
        else:
            obs = env.reset()

    return idx, net, failed_configs

    # plt.show()


def generateExpertTrajectory(env, r="", maze=None):
    if maze is not None:
        obs = env.reset(grid_string=maze)
    elif r == "g":
        env.set_goal(random_goal(env))
        obs = env.reset()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
    else:
        obs = env.reset()

    policy = solve_maze(env.grid_string)

    cur_trajectory = Trajectory()
    while True:
        action = policy[obs[0], obs[1]]

        obs_old = obs

        obs, reward, term, trunc, info = env.step(action)

        cur_trajectory.add_transition(list(obs_old), action, obs)

        env.render()

        if term or trunc:
            break

    return cur_trajectory
