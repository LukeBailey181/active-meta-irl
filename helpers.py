import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.colors import ListedColormap

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
###   solve_maze: a function given by                                         ###
###                grid_string (N, N) X  ---> Optimal Moves (N, N)            ###
###   visualize_optimal_moves: a function given by                            ###
###                Optimal Moves (N, N) ---> Heatmap + Arrows of the maze     ###
#################################################################################

# Dictionaries with all the possible actions/keys/tuples
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

tuple_to_key = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


# Returns the transition matrix for a given maze
#     T[i * board_size + j][action][k * board_size + l] = P((k,l) | (i,j), action)
def get_transition_matrix(grid_string):
    N = grid_string.shape[0]
    T = np.zeros((N**2, 4, N**2))

    for i in range(N):
        for j in range(N):
            # If a wall or goal, it doesn't matter---just have stay in the same place WP 1
            if grid_string[i, j] == 1 or grid_string[i, j] == 3:
                print(i * N + j)
                T[i * N + j][:, i * N + j] = 1
                continue

            # Otherwise, the agent might be here
            for action in range(4):
                # Try to go right
                if action == 0:
                    print(i, j)
                    if grid_string[i, j + 1] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, i * N + j + 1] = 1
                elif action == 1:
                    if grid_string[i + 1, j] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, (i + 1) * N + j] = 1
                elif action == 2:
                    if grid_string[i, j - 1] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, i * N + j - 1] = 1
                elif action == 3:
                    if grid_string[i - 1, j] == 1:
                        T[i * N + j, action, i * N + j] = 1
                    else:
                        T[i * N + j, action, (i - 1) * N + j] = 1
    return T


# Shows the transition probabilities for a given state given transition matrix T
def visualize_transition(T, state):
    N = int(np.sqrt(T.shape[0]))
    x, y = state
    fig, axs = plt.subplots(2, 2)

    # For each action
    for i in range(4):
        t_grid = T[x * N + y][i].reshape((N, N))
        # Plot the transition probabilities as a heatmap
        axs[i // 2, i % 2].imshow(t_grid, cmap="hot", interpolation="nearest")
        axs[i // 2, i % 2].set_title(f"Action {action_to_key[i]}")
        axs[i // 2, i % 2].set_xlabel("x")
        axs[i // 2, i % 2].set_ylabel("y")

        # add a red dot to the current state
        axs[i // 2, i % 2].scatter(y, x, c="r", s=40)

    plt.show()


# Returns a matrix of optimal moves for a given maze at each state
def solve_maze(maze):
    maze = maze.T
    goal = np.argwhere(maze == 3)[0]
    maze = (maze == 1).astype(int)

    optimal_moves = np.zeros(maze.shape, dtype=int) - 1
    # The moves at the goal + walls don't matter
    optimal_moves[goal[0]][goal[1]] = key_to_action["right"]
    optimal_moves[maze == 1] = 0

    # Work back through all squares to get the optimal moves
    current = deque([goal])

    while current:
        x, y = current.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Check if the neighbor is valid
            new_x, new_y = x + dx, y + dy
            if (
                0 <= new_x < maze.shape[0]
                and 0 <= new_y < maze.shape[1]
                and optimal_moves[new_x][new_y] == -1
            ):
                # If so, add it to the search queue and set its optimal move
                current.append((new_x, new_y))
                optimal_moves[new_x][new_y] = key_to_action[tuple_to_key[(-dx, -dy)]]

    return optimal_moves.T


# Visualizes the optimal moves for a given maze
def visualize_optimal_moves(maze, optimal_moves, save=False):
    # Plot the maze so that walls are black, empty spaces are white, the goal is green, and the start is blue
    cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])

    # Visualize the maze
    plt.imshow(maze.T, cmap=cmap, interpolation="nearest")
    plt.title("Optimal Maze Actions")
    N = maze.shape[0]

    # Parameters for the arrows
    hw = 0.5
    hl = 0.55
    so = 0.2
    l = 0.005
    col = "red"

    # Visualize the optimal moves
    for i in range(N):
        for j in range(N):
            if maze[i][j] == 1:
                continue
            if optimal_moves[i][j] == key_to_action["left"]:
                plt.arrow(
                    i + so, j, -l, 0, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["right"]:
                plt.arrow(
                    i - so, j, l, 0, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["up"]:
                plt.arrow(
                    i, j + so, 0, -l, head_width=hw, head_length=hl, fc=col, ec=col
                )
            elif optimal_moves[i][j] == key_to_action["down"]:
                plt.arrow(
                    i, j - so, 0, l, head_width=hw, head_length=hl, fc=col, ec=col
                )
    # plt.show()
    # Save the image
    if save:
        plt.savefig("optimal_directions.png")
    else:
        plt.show()


####################################
#####        Optimizes       #######
####################################

"""
Generic stochastic gradient-ascent based optimizers.

Due to the MaxEnt IRL objective of maximizing the log-likelihood instead of
minimizing a loss function, all optimizers in this module are actually
stochastic gradient-ascent based instead of the more typical descent.
"""



class Optimizer:
    """
    Optimizer base-class.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
    """
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments are optimizer-specific.
        """
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        """
        Create a new wrapper for this optimizer which normalizes the
        gradient before each step.

        Returns:
            An Optimizer instance wrapping this Optimizer, normalizing the
            gradient before each step.

        See also:
            `class NormalizeGrad`
        """
        return NormalizeGrad(self, ord)


class Sga(Optimizer):
    """
    Basic stochastic gradient ascent.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`
            taking the step number as parameter and returning a learning
            rate as result.
            See also `linear_decay`, `power_decay` and `exponential_decay`.

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
        lr: The learning-rate as specified in the __init__ function.
        k: The number of steps run since the last reset.
    """
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters += lr * grad


class ExpSga(Optimizer):
    """
    Exponentiated stochastic gradient ascent.

    The implementation follows Algorithm 10.5 from B. Ziebart's thesis
    (2010) and is slightly adapted from the original algorithm provided by
    Kivinen and Warmuth (1997).

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`
            taking the step number as parameter and returning a learning
            rate as result.
            See also `linear_decay`, `power_decay` and `exponential_decay`.
        normalize: A boolean specifying if the the parameters should be
            normalized after each step, as done in the original algorithm by
            Kivinen and Warmuth (1997).

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
        lr: The learning-rate as specified in the __init__ function.
        k: The number of steps run since the last reset.
    """
    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad)

        if self.normalize:
            self.parameters /= self.parameters.sum()


class NormalizeGrad(Optimizer):
    """
    A wrapper wrapping another Optimizer, normalizing the gradient before
    each step.

    For every call to `step`, this Optimizer will normalize the gradient and
    then pass the normalized gradient on to the underlying optimizer
    specified in the constructor.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        opt: The underlying optimizer to be used.
        ord: The order of the norm to be used for normalizing. This argument
            will be direclty passed to `numpy.linalg.norm`.
    """
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        This will call the underlying optimizer with the normalized
        gradient.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments depend on the underlying optimizer.
        """
        return self.opt.step(grad / np.linalg.norm(grad, self.ord), *args, **kwargs)


def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    """
    Linear learning-rate decay.

    Creates a function `(k: Integer) -> learning_rate: Float` returning the
    learning-rate in dependence on the current number of iterations. The
    returned function can be expressed as

        learning_rate(k) = lr0 / (1.0 + decay_rate * floor(k / decay_steps))

    Args:
        lr0: The initial learning-rate.
        decay_rate: The decay factor.
        decay_steps: An integer number of steps that can be used to
            staircase the learning-rate.

    Returns:
        The function giving the current learning-rate in dependence of the
        current iteration as specified above.
    """
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr


def power_decay(lr0=0.2, decay_rate=1.0, decay_steps=1, power=2):
    """
    Power-based learning-rate decay.

    Creates a function `(k: Integer) -> learning_rate: Float` returning the
    learning-rate in dependence on the current number of iterations. The
    returned function can be expressed as

        learning_rate(k) = lr0 / (1.0 + decay_rate * floor(k / decay_steps))^power

    Args:
        lr0: The initial learning-rate.
        decay_rate: The decay factor.
        decay_steps: An integer number of steps that can be used to
            staircase the learning-rate.
        power: The exponent to use for decay.

    Returns:
        The function giving the current learning-rate in dependence of the
        current iteration as specified above.
    """
    def _lr(k):
        return lr0 / (decay_rate * np.floor(k / decay_steps) + 1.0)**power

    return _lr


def exponential_decay(lr0=0.2, decay_rate=0.5, decay_steps=1):
    """
    Exponential learning-rate decay.

    Creates a function `(k: Integer) -> learning_rate: Float` returning the
    learning-rate in dependence on the current number of iterations. The
    returned function can be expressed as

        learning_rate(k) = lr0 * e^(-decay_rate * floor(k / decay_steps))

    Args:
        lr0: The initial learning-rate.
        decay_rate: The decay factor.
        decay_steps: An integer number of steps that can be used to
            staircase the learning-rate.

    Returns:
        The function giving the current learning-rate in dependence of the
        current iteration as specified above.
    """
    def _lr(k):
        return lr0 * np.exp(-decay_rate * np.floor(k / decay_steps))

    return _lr


class Initializer:
    """
    Base-class for an Initializer, specifying a strategy for parameter
    initialization.
    """
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Create an initial set of parameters.

        Args:
            shape: The shape of the parameters.

        Returns:
            An initial set of parameters of the given shape, adhering to the
            initialization-strategy described by this Initializer.
        """
        raise NotImplementedError

    def __call__(self, shape):
        """
        Create an initial set of parameters.

        Note:
            This function simply calls `self.initialize(shape)`.

        Args:
            shape: The shape of the parameters.

        Returns:
            An initial set of parameters of the given shape, adhering to the
            initialization-strategy described by this Initializer.
        """
        return self.initialize(shape)


class Uniform(Initializer):
    """
    An Initializer, initializing parameters according to a specified uniform
    distribution.

    Args:
        low: The minimum value of the distribution.
        high: The maximum value of the distribution

    Attributes:
        low: The minimum value of the distribution.
        high: The maximum value of the distribution
    """
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def initialize(self, shape):
        """
        Create an initial set of uniformly random distributed parameters.

        The parameters of the distribution can be specified in the
        constructor.

        Args:
            shape: The shape of the parameters.

        Returns:
            An set of initial uniformly distributed parameters of the given
            shape.
        """
        return np.random.uniform(size=shape, low=self.low, high=self.high)


class Constant(Initializer):
    """
    An Initializer, initializing parameters to a constant value.

    Args:
        value: Either a scalar value or a function in dependence on the
            shape of the parameters, returning a scalar value for
            initialization.
    """
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        """
        Create set of parameters with initial fixed value.

        The scalar value used for initialization can be specified in the
        constructor.

        Args:
            shape: The shape of the parameters.

        Returns:
            An set of constant-valued parameters of the given shape.
        """
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value