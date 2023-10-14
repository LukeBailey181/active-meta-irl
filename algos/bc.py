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


def BehaviorCloning(
    train_dataset,
    test_dataset=None,
    env=None,
    epochs=2000,
    epsilon=0.1,
    batch_size=32,
    lr=0.001,
    eval_freq=None,
    save_weights=False,
    r="",
):
    state_size = train_dataset[0][0].shape[0]

    # Define the network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 4)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

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

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} / {epochs}")
        # Training loop
        epoch_train_lossses = []
        epoch_train_accs = []
        for i, (state, action) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(state)
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

                        num_runs = 25

                        for i in range(num_runs):
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

    save_string = f"bc_{env.grid_string.shape[0]}x{env.grid_string.shape[0]}_r_{r}_samples_{train_dataset.__len__()}"

    # Save the model
    if save_weights:
        torch.save(net.state_dict(), "logs/bc/models/" + save_string + ".pt")

    plt.savefig("logs/bc/" + save_string + ".png")

    # Save the data as a numpy array
    np.savez(
        "logs/bc/" + save_string + ".npz",
        train_losses=train_losses,
        test_losses=test_losses,
        train_accs=train_accs,
        test_accs=test_accs,
        eval_reward=eval_reward,
        logged_epochs=logged_epochs,
    )

    plt.show()


def generateExpertDataset(env, r="", num_train_samples=500, num_test_samples=100):
    num_samples = (
        num_train_samples + num_test_samples
        if num_test_samples is not None
        else num_train_samples
    )
    # Collect expert data
    states = np.zeros((num_samples, 4))
    actions = np.zeros((num_samples, 1))

    if r == "g":
        env.set_goal(random_goal(env))
        obs = env.reset()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
    else:
        obs = env.reset()

    policy = solve_maze(env.grid_string)

    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Collecting sample {i} of {num_samples}")
        action = policy[obs[0], obs[1]]

        states[i] = obs
        actions[i] = action

        obs, reward, term, trunc, info = env.step(action)

        env.render()
        if term or trunc:
            if r == "g":
                env.set_goal(random_goal(env))
                obs = env.reset()
            elif r == "m":
                obs = env.reset(grid_string=generate_maze(env.board_size))
            else:
                obs = env.reset()

            policy = solve_maze(env.grid_string)

            # Check that the policy is valid
            try:
                assert np.all(policy != -1)
            except:
                print("INVALID POLICY !!!!! THIS SHOULD NEVER HAPPEN")
                print(policy)
                print(env.grid_string)
                break

    train_dataset = (states[0:num_train_samples], actions[0:num_train_samples])
    test_dataset = (states[num_train_samples:], actions[num_train_samples:])

    if num_test_samples is None:
        test_dataset = None

    return train_dataset, test_dataset
