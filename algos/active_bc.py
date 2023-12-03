import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from maze_env import MutableMaze
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from goal_setters import random_goal
from mazes import *
import os
import yaml
from helpers import Net, ConvNet
from maze_env import Trajectory
from algos.acquisition_funcs import failed_min_max_acquisition
from helpers import generateExpertTrajectory, generate_maze



def active_bc(env=None, budget=100, config=None):

    r = config["base"]["randomize"]
    #if r == "g":
    #    goal_randomized_bc_al(env, budget, config)
    #elif r == "m":
    #    maze_randomized_bc_al(env, budget, config)
    #else:
    #    return

    if r in ["g", "m"]:
        randomized_bc_al(env, budget, config)
    else: 
        raise NotImplementedError(f"Randomization type {r} not implemented")


def randomized_bc_al(env, budget, config):
    """
    Active BC with randomization for goal and maze using heuristic over failed mazes
    """ 

    # ----- Setup for active learning ----- #
    cur_mazes = []
    max_heuristics = []
    acquisition_func_name = config["al"]["acquisition_func_name"]
    batch_size = config["al"]["batch_size"]
    num_trajectories_sampled = 0
    states = []
    actions = []

    r = config["base"]["randomize"]

    init_trajectories = []
    num_traj_to_sample = min(batch_size, budget)
    for i in range(num_traj_to_sample):
        # Generate initial random mazes
        if r == "m":
            cur_mazes.append(np.copy(generate_maze(env.board_size)))
        elif r == "g":
            goal_location = random_goal(env)
            env.set_goal(goal_location)
            cur_mazes.append(np.copy(env.grid_string))

            #print("\nDEBUG")
            #print(f"goals: {goal_location}\n")
            #cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])
            #plt.imshow(cur_mazes[-1].T, cmap=cmap)
            #plt.show()

        init_trajectories.append(
            generateExpertTrajectory(env, r=config["base"]["randomize"], maze=cur_mazes[i])
        )
        num_trajectories_sampled += 1

    # Generate the initial dataset
    state_action_tuples = [traj.transitions() for traj in init_trajectories]

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

    # ----- Active learning loop ----- #
    while num_trajectories_sampled < budget:
        print(f"\nTraining on {num_trajectories_sampled} expert samples")
        print(f"DEBUG\bnum_mazes = {len(cur_mazes)}")

        # Calculate how many trajectories we want
        num_traj_to_sample = min(batch_size, budget - num_trajectories_sampled)

        num_trajectories_sampled += num_traj_to_sample

        # Query aquisition function and update dataset
        if acquisition_func_name == "failed_min_max":
            new_states, new_actions, cur_mazes, info = failed_min_max_acquisition(
                env=env,
                config=config,
                num_trajectories=num_traj_to_sample,
                cur_mazes=cur_mazes,
                failed_configs=failed_configs,
            )
        else:
            raise NotImplementedError(f"Acquisition function {acquisition_func_name} not implemented")
        
        #breakpoint()
        states += new_states
        actions += new_actions

        # Run BC on the updated dataset
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

        # Save model and state and action datasets
        save_model(net, config, num_trajectories_sampled, np_states, np_actions)
 
    assert(num_trajectories_sampled == budget)
    # ----- SAVE MODEL AND PLOT RESULTS ----- #

    # Plot the max heuristics
    #plt.plot(max_heuristics)
    #plt.xlabel("Iteration")
    #plt.ylabel("Max Heuristic")
    #plt.title("Max Heuristic vs. Iteration")
    #plt.savefig(
    #    f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/max_heuristics.png"
    #)

    return states, actions 

def find_goal(mazes):

    goals = []
    for maze in mazes:
        goals.append(np.argwhere(maze == 3)[0])
        print(np.argwhere(maze == 3)[0])

    for i in range(len(goals)):
        for j in range(i+1, len(goals)): 
            if (goals[i] == goals[j]).all():
                print(f"DUP {goals[i]}")
    



def save_model(net, config, budget, states=None, actions=None):
    # If logs/bc_al/al_model_{budget} doesn't exist, create it
    if not os.path.exists(
        f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}"
    ):
        os.makedirs(f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}")

    torch.save(
        net.state_dict(),
        f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/bc_al_model_samples_{budget}.pt",
    )

    if states is not None and actions is not None:
        np.save(f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/states_samples_{budget}.npy", states)
        np.save(f"logs/{config['base']['save_dir']}/bc_al_model_samples_{budget}/actions_samples_{budget}.npy", actions)

def maze_randomized_bc_al_DEP(env, budget, config):
    mazes = []
    max_heuristics = []
    trajectories = []
    heuristic = config["al"]["heuristic"]
    assert(heuristic is not None)

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

        ### ABOVE HERE IS AQUISITION FUNCTION

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

def goal_randomized_bc_al_DEP(env, budget, config):
    mazes = []

    r = config["base"]["randomize"]
    heuristic = config["al"]["heuristic"]
    assert(heuristic is not None)

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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on Apple M1 (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    eps = 1e-8

    if network == "fc":
        if net is None:
            net = Net(state_size)
            net.to(device)
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
            #if torch.cuda.is_available():
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

    print("Final Train Accs:")
    print(train_accs[-1])
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
                    #failed_configs.append(np.copy(env.grid_string.T))
                    failed_configs.append(np.copy(env.grid_string))
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
