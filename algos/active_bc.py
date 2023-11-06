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
from helpers import Net, ConvNet
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
    r = config["base"]["randomize"]

    if r == "g":
        goal_randomized_bc_al(env, budget, config)
    elif r == "m":
        maze_randomized_bc_al(env, budget, config)
    else:
        return


def maze_randomized_bc_al(env, budget, config):
    mazes = []
    max_heuristics = []
    trajectories = []

    B = 100

    r = config["base"]["randomize"]

    for i in range(B):
        mazes.append(np.copy(generate_maze(env.board_size)))
        trajectories.append(
            generateExpertTrajectory(env, r=config["base"]["randomize"], maze=mazes[i])
        )

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
            heuristics = np.zeros(len(failed_configs))
            max_heuristic = 0
            max_heuristic_idx = 0
            for t in range(len(failed_configs)):
                cur_maze_heuristics = np.zeros(len(mazes))
                maze_count = 0
                for maze in mazes:
                    cur_maze_heuristics[maze_count] = heuristic(maze, failed_configs[t])
                    maze_count += 1
                min_heuristic = np.min(cur_maze_heuristics)

                heuristics[t] = min_heuristic

            max_heuristic = np.max(heuristics)

            print(f"Max heuristic: {max_heuristic}")
            max_heuristics.append(max_heuristic)

            num_left = min(B, budget - len(trajectories))

            num_avail = len(failed_configs)

            if num_avail >= num_left:
                # Find the num_left best heuristics
                best_heuristics = np.argsort(heuristics)[-num_left:]
                # Append the corresponding configs to mazes
                for h in best_heuristics:
                    mazes.append(np.copy(failed_configs[h]))
                    trajectories.append(
                        generateExpertTrajectory(
                            env, r=config["base"]["randomize"], maze=failed_configs[h]
                        )
                    )
                    for state, action, _ in trajectories[-1].transitions():
                        states.append(state)
                        actions.append(action)
            else:
                # Append all of the configs to mazes
                for h in range(num_avail):
                    mazes.append(np.copy(failed_configs[h]))
                    trajectories.append(
                        generateExpertTrajectory(
                            env, r=config["base"]["randomize"], maze=failed_configs[h]
                        )
                    )
                    for state, action, _ in trajectories[-1].transitions():
                        states.append(state)
                        actions.append(action)
                # Append random mazes to mazes
                for h in range(num_left - num_avail):
                    mazes.append(np.copy(generate_maze(env.board_size)))
                    trajectories.append(
                        generateExpertTrajectory(
                            env, r=config["base"]["randomize"], maze=mazes[-1]
                        )
                    )
                    for state, action, _ in trajectories[-1].transitions():
                        states.append(state)
                        actions.append(action)
        else:
            # Append random mazes to mazes
            num_left = min(B, budget - len(trajectories))
            for h in range(num_left):
                mazes.append(np.copy(generate_maze(env.board_size)))
                trajectories.append(
                    generateExpertTrajectory(
                        env, r=config["base"]["randomize"], maze=mazes[-1]
                    )
                )
                for state, action, _ in trajectories[-1].transitions():
                    states.append(state)
                    actions.append(action)

        np_states = np.array(states)
        np_actions = np.array(actions)

        idx, net, failed_configs = BehaviorCloning(
            train_dataset=(np_states, np_actions),
            test_dataset=None,
            env=env,
            config=config,
            net=None,
            idx=(idx + 1),
        )

    # Save the final model

    # If logs/bc_al/al_model_{budget} doesn't exist, create it
    if not os.path.exists(
        f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}"
    ):
        os.makedirs(f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}")

    torch.save(
        net.state_dict(),
        f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/bc_al_model_samples_{budget}.pt",
    )

    # Plot the max heuristics
    plt.plot(max_heuristics)
    plt.xlabel("Iteration")
    plt.ylabel("Max Heuristic")
    plt.title("Max Heuristic vs. Iteration")
    plt.savefig(
        f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/max_heuristics.png"
    )


def goal_randomized_bc_al(env, budget, config):
    mazes = []

    r = config["base"]["randomize"]

    env.set_goal(random_goal(env))
    init_maze = np.copy(env.grid_string.T)

    max_heuristics = []

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
            cur_maze = np.copy(env.grid_string.T)

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
            net=None,
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
    budget=None,
    idx=0,
):
    r = config["base"]["randomize"]
    epochs = config["bc"]["num_epochs"]
    batch_size = config["bc"]["batch_size"]
    lr = config["bc"]["learning_rate"]
    save_weights = config["bc"]["save_weights"]
    eval_freq = config["bc"]["eval_freq"]
    num_eval_runs = config["bc"]["num_eval_runs"]
    network = config["base"]["network"]

    state_size = train_dataset[0][0].shape[0]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    eps = 1e-8

    if network == "fc":
        if net is None:
            net = Net(state_size)
    elif network == "cnn":
        if net is None:
            net = ConvNet(config["base"]["size"])
            net.to(device)

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
            if torch.cuda.is_available():
                state = state.to(device)
                action = action.to(device)

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

    failed_configs = []

    obs = env.reset()

    if r == "g":
        maze_bin = (env.grid_string == 1) + (env.grid_string == 2)
        # All possible goals
        goals = np.argwhere(maze_bin == 0)
        goal_idx = 1
        env.set_goal(goals[0])
        obs = env.reset()
        num_trials = len(goals) - 1
    elif r == "m":
        obs = env.reset(grid_string=generate_maze(env.board_size))
        num_trials = 100 * 5
    else:
        obs = env.reset()

    for i in range(num_trials):
        reward_sum = 0
        while True:
            obs_tensor = torch.tensor(obs).float().to(device).unsqueeze(0)
            action = net(obs_tensor).argmax().item()
            obs, reward, term, trunc, info = env.step(int(action))
            env.render()
            if term or trunc:
                if reward == 0:
                    failed_configs.append(np.copy(env.grid_string.T))
                break

        if r == "g":
            env.set_goal(goals[goal_idx])
            goal_idx += 1
            obs = env.reset()
        elif r == "m":
            obs = env.reset(grid_string=generate_maze(env.board_size))
        else:
            obs = env.reset()

    return idx, net, failed_configs


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
