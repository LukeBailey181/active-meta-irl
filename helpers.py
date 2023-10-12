import numpy as np
import matplotlib.pyplot as plt

#################################################################################
###  This file contains a number of helper functions, including:              ###
###                                                                           ###                          
###   key_to_action: a dictionary mapping strings to actions                  ###
###   action_to_key: a dictionary mapping actions to strings                  ###
###                                                                           ###
###   get_transition_matrix: a function given by                              ###         
###                grid_string (N, N) ---> Transition Matrix (N^2, A, N^2)    ###
###   visualize_transition: a function given by                               ###
###                Transition Matrix (N^2, A, N^2) X State (x,y) ---> Heatmap ###
#################################################################################

# Dictionary with all the possible actions
key_to_action = {
    "left": 2,
    "right": 0,
    "up": 3,
    "down": 1,
}

action_to_key = {
    2: "left",
    0: "right",
    3: "up",
    1: "down",
}

# Returns the transition matrix for a given maze
#     T[i * board_size + j][action][k * board_size + l] = P((k,l) | (i,j), action)
def get_transition_matrix(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N**2, 4, N**2))

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i,j] == 1 or grid_string[i,j] == 3:
                print(i*N + j)
                T[i * N + j][:,i * N + j] = 1
                continue
            
            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    print(i,j)
                    if grid_string[i,j+1] == 1:
                        T[i*N+j, action, i*N+j] = 1
                    else:
                        T[i*N+j, action, i*N+j+1] = 1
                elif action == 1:
                    if grid_string[i+1,j] == 1:
                        T[i*N+j, action, i*N+j] = 1
                    else:
                        T[i*N+j, action, (i+1)*N+j] = 1
                elif action == 2:
                    if grid_string[i,j-1] == 1:
                        T[i*N+j, action, i*N+j] = 1
                    else:
                        T[i*N+j, action, i*N+j-1] = 1
                elif action == 3:
                    if grid_string[i-1,j] == 1:
                        T[i*N+j, action, i*N+j] = 1
                    else:
                        T[i*N+j, action, (i-1)*N+j] = 1
    return T

# Shows the transition probabilities for a given state given transition matrix T
def visualize_transition(T, state):
    N = int(np.sqrt(T.shape[0]))
    x, y = state
    fig, axs = plt.subplots(2, 2)

    # For each action
    for i in range(4):
        t_grid = T[x*N+y][i].reshape((N,N))
        # Plot the transition probabilities as a heatmap
        axs[i//2, i%2].imshow(t_grid, cmap='hot', interpolation='nearest')
        axs[i//2, i%2].set_title(f"Action {action_to_key[i]}")
        axs[i//2, i%2].set_xlabel("x")
        axs[i//2, i%2].set_ylabel("y")
        
        # add a red dot to the current state
        axs[i//2, i%2].scatter(y, x, c='r', s=40)

    plt.show()
