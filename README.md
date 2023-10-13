# Active Meta-IRL

## Intructions for Install

1. Create a new conda environment with Python 3.8

```
conda create -n metairl python=3.8

conda activate metairl
```

2. Install dependencies

```
pip install minigrid gym pygame matplotlib numpy
```

## Intructions for Use

#### Running Experiments

```maze_runner``` takes several arguments which specify the experiment:

* ```--control (-c)```: The method of control, from ['expert', 'manual', 'random', 'policy']
* ```--randomize (-r)```: The degree of randomization, from ['m', 'g', ''] for randomized maze, goal, and none.
* ```---size (-s)```: If maze is randomized, the size of the generated maze.
* ```---maze (-m)```: If goal or none are randomized, the default maze configuration.


So, for example, to show the expert policy run on a random maze of size 20, we may run

```
python maze_runner.py -c expert -r m -s 20
```

If we want to train an RL policy on some maze of size 10, randomizing goal only, we may run

```
python maze_runner.py -c policy -r g -s 10
```

#### File Structure

```maze_env``` Contains the environment for creating an running policies in mazes.

```maze_runner``` Contains code for executing automatic policies or applying manual control

```manual_conroller``` contains our manual control system, for debugging and visualization (**and providing expert samples????**)

```goal_setters``` contains methods for changing the goal, given an environment as input.

```helpers``` contains helper methods for performing common computations.

```mazes``` contains both fixed mazes and functions for generating novel mazes.
