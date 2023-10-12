# Active Meta-IRL

## Intructions for Install

1. Create a new conda environment with Python 3.8

```
conda create -n metairl python=3.8
```

2. Install dependencies

```
pip install minigrid gym pygame matplotlib numpy
```

## Intructions for Use

```maze_env``` Contains the environment for creating an running policies in mazes.

```maze_runner``` Contains code for executing automatic policies or applying manual control

```manual_conroller``` contains our manual control system, for debugging and visualization (**and providing expert samples????**)

```goal_setters``` contains methods for changing the goal, given an environment as input.

```helpers``` contains helper methods for performing common computations.

```mazes``` contains both fixed mazes and functions for generating novel mazes.
