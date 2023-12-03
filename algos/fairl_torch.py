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
from helpers import (
    Net,
    ConvNet,
    get_transition_matrix,
    visualize_reward,
    get_transition_states,
    get_transition_deltas,
    maze_map,
    generate_maze
)
from maze_env import Trajectory


# Define the network
class ValNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        output = torch.sigmoid(x) * 10
        return output


class ValCNN(nn.Module):
    def __init__(self, state_size):
        self.fc_size = 32 * state_size**2
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


# Calculates the negative log-likelihood of the correct actions under the network
class LikelihoodLoss(nn.Module):
    def __init__(self, net, b, N):
        super(LikelihoodLoss, self).__init__()
        self.net = net
        self.b = b
        self.N = N

    def forward(self, states, actions):
        likelihood = 0

        l = actions.shape[0]

        # Caclulate the log-likelihood of each correct action
        for i, a in enumerate(actions):
            if states[i][0] == 8.0 and states[i][1] == 6.0 and False:
                print("\n\n Running eval")
                self.net.readout = True
            likelihood -= self.b * self.net.Q(states[i], int(a)) - torch.log(
                torch.sum(torch.exp(self.b * self.net.Q_vector(states[i])))
            )

            self.net.readout = False

        # Normalize it to find the average likelihood for the sample
        likelihood = likelihood / l

        return likelihood


# This network wraps our NN to calculate Q, V, and R
class FAIRLNet:
    def __init__(
        self,
        net_type,
        net,
        config,
        env,
        r,
    ):
        self.net = net
        self.net_type = net_type
        self.config = config
        self.env = env
        self.grid_string = self.env.grid_string.copy()
        self.P = get_transition_states(self.grid_string.T)
        self.N = self.grid_string.shape[0]
        self.gamma = 0.2
        self.readout = False
        self.rand = r

    # Change the maze
    def set_maze(self, m):
        self.grid_string = m.copy()
        # self.env.set_grid_string(m.copy())
        # print(self.grid_string.T.copy())
        self.P = get_transition_states(self.grid_string.T.copy())

    # Set the network to eval mode
    def eval(self):
        self.net.eval()

    # Set the network to train mode
    def train(self):
        self.net.train()

    # Find the next state given s and a, encoded using the FCN input space
    def next_state_fcn(self, s, a):
        s_cp = s.clone().tolist()
        next_state = self.P[int(s_cp[0]), int(s_cp[1])][a]
        if self.readout and False:
            print(self.P.shape)
            print(f"A: {a}")
            print(f"Next state: {next_state}")
        s_prime = torch.tensor([next_state[0], next_state[1], s_cp[2], s_cp[3]])
        return s_prime

    # Calculate the optimal Q function for a given state and action.
    def Q(self, s, a, m=None):
        assert type(a) == int

        # Make sure that the model of the maze is up to date
        if m is not None:
            print("Does this ever happen?")
            self.set_maze(m)

        if self.rand == "g":
            s_cp = s.clone().tolist()
            # self.env.set_goal([s_cp[3], s_cp[2]])
            m = self.grid_string.copy()
            current_goal_pos = np.argwhere(m == 3)[0]
            m[current_goal_pos[0]][current_goal_pos[1]] = 0
            m[int(s_cp[2])][int(s_cp[3])] = 3
            if self.readout:
                print(f"Maze position is {s_cp}")
                plt.imshow(m.T, cmap=maze_map)
                plt.show()
            self.set_maze(m)

        # Grab the state that f will transition to under a
        next_state = self.next_state_fcn(s, a)

        # Normalize the state to lie in [-1, 1]
        next_state_norm = (next_state - self.N / 2) / self.N

        # Push to cuda
        next_state_norm = next_state_norm.float().cuda()

        # Find the Q of the next state
        # This is a special case of the expected sum of f from the paper, since the dynamics are deterministic
        Q = self.net(next_state_norm)

        # if self.readout:
        #     print("In Q function")
        #     print(f"State is {s}")
        #     print(f"I take action {a}")
        #     print(f"next state is {next_state}")
        #     print(f"Net output is {Q}")

        # Make sure that the network didn't overflow
        if Q[0].isnan():
            print("Nan found!")
            print(f"S: {s}")
            print(f"A : {a}")
            print(f"next_state: {next_state}")

        return Q

    # Find Q for all actions in a state s
    def Q_vector(self, s, m=None):
        Q_vector = torch.zeros(4).float().cuda()

        for a in range(4):
            Q_vector[a] = self.Q(s, a)

        return Q_vector

    # Find the optimal value function at a state
    def V(self, s, m=None):
        # self.readout = True
        # print(s)
        # print(s.shape)
        return torch.max(self.Q_vector(s, m))

    # Calculate the reward at a state
    def r(self, s, m=None):
        # Scale S to work for network input
        s_scaled = (s - self.N / 2) / self.N

        output = self.net(s_scaled) - self.gamma * self.V(s, m)
        return output

    # Calculate the reward at all points in a maze
    def get_maze_reward(self, m=None):
        # Update the maze if necessary
        if m is not None:
            self.set_maze(m)

        # Initialize r
        r = np.zeros((self.N, self.N))

        # Find the maze's goal location
        goal_pos = np.argwhere(m == 3)[0]

        # For each point in the maze, calculate the reward
        for i in range(self.N):
            for j in range(self.N):
                # Grab the reward at that point
                r[i, j] = self.r(
                    torch.tensor([i, j, goal_pos[0], goal_pos[1]]).float().cuda()
                )
        return r


# Perform one epoch of training
def train_step(train_loader, optimizer, criterion, f_net):
    # Training loop
    epoch_train_lossses = []
    epoch_train_accs = []

    # Perform batched SGD
    for i, (state, action) in enumerate(train_loader):
        optimizer.zero_grad()

        # Put everything on GPU if possible
        if torch.cuda.is_available():
            state = state.cuda()
            action = action.cuda()

        # Fix possible shape error for last batch
        if state.shape[0] != 1:
            action = action.squeeze()

        # Evaluate the likelihood of the action
        loss_act = criterion(state, action)
        # Take a step
        loss_act.backward()
        optimizer.step()

        # Calculate train loss
        with torch.no_grad():
            # Find the greedy action for the current policy
            output = torch.zeros((state.shape[0],), dtype=torch.long)
            for i, s in enumerate(state):
                output[i] = f_net.Q_vector(s).argmax().item()

            # The number of elements in the batch
            total = action.size(0)

            action = action.cpu()

            # Find how many were correct
            correct = (output == action).sum().item()
            acc = correct / total

        epoch_train_lossses.append(loss_act.item())
        epoch_train_accs.append(acc)

    return np.mean(epoch_train_lossses), np.mean(epoch_train_accs)


# Perform one epoch of evaluation
def eval_step(test_dataset, test_loader, f_net, criterion, env, num_eval_runs, r):
    # Evaluate data in a supervised manner
    if test_dataset is None:
        print("No test dataset provided, skipping evaluation")
        epoch_test_accs = None
        epoch_test_lossses = None
    else:
        epoch_test_lossses = []
        epoch_test_accs = []
        for state, action in test_loader:
            # Put everything on the GPU
            if torch.cuda.is_available():
                state = state.cuda()
                action = action.cuda()

            # Grab each element of the output
            output = torch.zeros((state.shape[0],), dtype=torch.long)
            Q_vecs = torch.zeros((state.shape[0], 4))
            for i, s in enumerate(state):
                output[i] = f_net.Q_vector(s).argmax().item()
                Q_vecs[i] = f_net.Q_vector(s)

            # Handle possible shape error
            if state.shape[0] != 1:
                action = action.squeeze()

            # Evaluate the likelihood of the action
            loss_act = criterion(state, action)

            # Batch size
            total = action.size(0)

            # _, predicted = torch.max(output.data, 1)
            action = action.cpu()
            correct = (output == action).sum().item()
            acc = correct / total

            # bad_actions = action[output != action]
            # bad_outputs = output[output != action]

            # print(f"In states \n {state[output != action]}")
            # print(f"I took the following actions : \n{bad_outputs}")
            # print(f"Instead of the following actions : \n{bad_actions}")
            # print(f"With Q vectors \n{Q_vecs[output != action]}")

            epoch_test_lossses.append(loss_act.item())
            epoch_test_accs.append(acc)

    # Evaluate the network via rollouts

    batch_reward = []
    # If the goal or maze is randomized, generate a new problem
    if r == "g":
        # goal_now = random_goal(env)
        # f_net.readout = True
        # plt.imshow(env.grid_string.T, cmap=maze_map)
        # plt.show()
        # env.set_goal([goal_now[1], goal_now[0]])
        # plt.imshow(env.grid_string.T, cmap=maze_map)
        # plt.show()

        goal = random_goal(env)
        # print(f"I set goal to {goal}")
        # self.env.set_goal([s_cp[3], s_cp[2]])
        m = env.grid_string.copy()
        current_goal_pos = np.argwhere(m == 3)[0]
        m[current_goal_pos[0]][current_goal_pos[1]] = 0
        m[goal[0]][goal[1]] = 3
        f_net.set_maze(m)
        env.set_grid_string(m)

        # plt.imshow(env.grid_string.T, cmap=maze_map)
        # plt.show()

        obs = env.reset()
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
        f_net.set_maze(env.grid_string)
    else:
        obs = env.reset()

    for i in range(num_eval_runs):
        reward_sum = 0
        while True:
            # Put it on the GPU if the network already is, and find the optimal action
            if torch.cuda.is_available():
                obs = torch.tensor(obs).float().cuda()
                action = f_net.Q_vector(obs).argmax().item()
            else:
                obs = torch.tensor(obs).float()
                action = f_net.Q_vector(obs).argmax().item()
                # print(f"Taking action {int(action)}")

            # Take the best action
            obs, reward, term, trunc, info = env.step(int(action))
            env.render()
            reward_sum += reward
            if term or trunc:
                break

        # If the goal or maze is randomized, generate a new problem
        if r == "g":
            goal = random_goal(env)
            # print(f"I set goal to {goal}")
            # self.env.set_goal([s_cp[3], s_cp[2]])
            m = env.grid_string.copy()
            current_goal_pos = np.argwhere(m == 3)[0]
            m[current_goal_pos[0]][current_goal_pos[1]] = 0
            m[goal[0]][goal[1]] = 3
            f_net.set_maze(m)
            env.set_grid_string(m)

            # plt.imshow(env.grid_string.T, cmap=maze_map)
            # plt.show()

            obs = env.reset()
        elif r == "m":
            obs = env.reset(grid_string=generate_maze(env.board_size))
            f_net.set_maze(env.grid_string)
        else:
            obs = env.reset()

        batch_reward.append(reward_sum)

    return epoch_test_accs, epoch_test_lossses, batch_reward


# Visualize and save results figures
def save_run(
    env,
    N,
    train_losses,
    test_dataset,
    logged_epochs,
    test_losses,
    train_accs,
    test_accs,
    eval_reward,
    batch_size,
    epochs,
    save_dir,
    save_weights,
    net,
    f_net,
    r,
):
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

    save_string = f"fairl_{env.grid_string.shape[0]}x{env.grid_string.shape[0]}_r_{r}_samples_{N}_batch_{batch_size}_epochs_{epochs}"

    # Make save_string directory
    if not os.path.exists(f"logs/{save_dir}/{save_string}"):
        os.makedirs(f"logs/{save_dir}/{save_string}")

    # Save the model
    if save_weights:
        torch.save(
            net.state_dict(), f"logs/{save_dir}/{save_string}/" + save_string + ".pt"
        )

    # Save the loss figure
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

    # Collect the final predicted reward on a set of mazes
    with torch.no_grad():
        if r == "m":
            random_maze = generate_maze(env.board_size)
            reward_vis = f_net.get_maze_reward(random_maze)
            reward_fig = visualize_reward(random_maze, reward_vis)
            plt.savefig(f"logs/{save_dir}/{save_string}/" + save_string + "_reward.png")
        elif r == "g":
            # Set a random goal
            for i in range(10):
                goal = random_goal(env)
                print(f"I set goal to {goal}")
                # self.env.set_goal([s_cp[3], s_cp[2]])
                m = env.grid_string.copy()
                current_goal_pos = np.argwhere(m == 3)[0]
                m[current_goal_pos[0]][current_goal_pos[1]] = 0
                m[goal[0]][goal[1]] = 3
                # f_net.set_maze(m)
                env.set_grid_string(m)

                reward_vis = f_net.get_maze_reward(env.grid_string)

                reward_fig = visualize_reward(env.grid_string, reward_vis)
                plt.title("Using discount factor gamma = .2")
                # Save the figure
                plt.savefig(
                    f"logs/{save_dir}/{save_string}/" + save_string + f"_reward_{i}.png"
                )
                plt.show()

                # reward_vis = f_net.get_maze_reward(env.grid_string)

                # reward_fig = visualize_reward(env.grid_string.T, reward_vis.T)
                # plt.title("Using discount factor gamma = .2 d")
                # # Save the figure
                # plt.savefig(
                #     f"logs/{save_dir}/{save_string}/" + save_string + f"_reward_d.png"
                # )
                # plt.show()

        else:
            reward_vis = f_net.get_maze_reward(env.grid_string)

            reward_fig = visualize_reward(env.grid_string.T, reward_vis)
            plt.title("Using discount factor gamma = .2")
            # Save the figure
            plt.savefig(
                f"logs/{save_dir}/{save_string}/" + save_string + f"_reward_b.png"
            )
            plt.show()

            reward_vis = f_net.get_maze_reward(env.grid_string.T)

            reward_fig = visualize_reward(env.grid_string.T, reward_vis)
            plt.title("Using discount factor gamma = .2")
            # Save the figure
            plt.savefig(
                f"logs/{save_dir}/{save_string}/" + save_string + f"_reward_f.png"
            )
            plt.show()


def RunFAIRL(
    train_dataset,
    test_dataset=None,
    env=None,
    config=yaml.load(open("configs/fairl/fairl_cfg.yaml", "r"), Loader=yaml.FullLoader),
    N=10,
):
    # Set hyperparameters
    r = config["base"]["randomize"]
    epochs = config["fairl"]["num_epochs"]
    batch_size = config["fairl"]["batch_size"]
    lr = config["fairl"]["learning_rate"]
    save_weights = config["fairl"]["save_weights"]
    eval_freq = config["fairl"]["eval_freq"]
    num_eval_runs = config["fairl"]["num_eval_runs"]
    network = config["base"]["network"]
    save_dir = config["base"]["save_dir"]
    size = config["base"]["size"]

    if torch.cuda.is_available():
        print("------------------Training on GPU------------------")
        device = torch.device("cuda:0")
    else:
        print(
            "------------------No device available, training on CPU------------------"
        )
        device = torch.device("cpu")

    if save_dir is None:
        save_dir = "fairl"

    # Initialize the network
    if network == "fc":
        state_size = train_dataset[0][0].shape[0]
        net = ValNet(state_size)
    elif network == "cnn":
        state_size = train_dataset[0][0].shape[0]
        net = ValCNN(state_size)

    net.to(device)
    # Create the network superclass
    f_net = FAIRLNet(network, net, config, env, r)

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

    print(f"State size: {size}")

    # Define the loss function
    criterion = LikelihoodLoss(f_net, b=5.0, N=size)

    # Define the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=True
    )

    print(f"Training on {len(train_dataset)} Samples.")

    # Train the network
    mod = eval_freq if eval_freq is not None else epochs // 10
    net.train()

    for epoch in range(epochs):
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} / {epochs}")

        # Training loop
        mean_train_losses, mean_train_accs = train_step(
            train_loader, optimizer, criterion, f_net
        )

        train_losses.append(mean_train_losses)
        train_accs.append(mean_train_accs)

        # Eval loop
        with torch.no_grad():
            if eval_freq is not None:
                if epoch % eval_freq == 0:
                    logged_epochs.append(epoch)
                    epoch_test_accs, epoch_test_lossses, batch_reward = eval_step(
                        test_dataset,
                        test_loader,
                        f_net,
                        criterion,
                        env,
                        num_eval_runs,
                        r,
                    )

                    if epoch_test_accs is not None:
                        test_losses.append(np.mean(epoch_test_lossses))
                        test_accs.append(np.mean(epoch_test_accs))

                    eval_reward.append(np.mean(batch_reward))

    save_run(
        env,
        N,
        train_losses,
        test_dataset,
        logged_epochs,
        test_losses,
        train_accs,
        test_accs,
        eval_reward,
        batch_size,
        epochs,
        save_dir,
        save_weights,
        net,
        f_net,
        r,
    )


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
        obs = env.reset(grid_string=maze.copy())
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
