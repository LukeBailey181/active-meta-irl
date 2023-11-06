# Active Meta-IRL

## Intructions for Install

1. Create a new conda environment with Python 3.8

```
conda create -n metairl python=3.8

conda activate metairl
```

2. Install dependencies

```
pip install minigrid gym pygame matplotlib numpy torch
```

## Running on GPU

If instead you want to run on GPU, we need to be a bit more fiddly to make sure versions work out.

1. Create a conda environment with Python 3.8

```
conda create -n goalirl_gpu python=3.8

conda activate goalirl_gpu
```

2. Install the necessary pytorch version for your CUDA version [find command here](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Install remaining dependencies

```
pip install minigrid gym pygame matplotlib numpy
```


## Intructions for Use

#### Running Experiments

```maze_runner``` takes several arguments which specify the experiment. These can either be specified by a config file or in the command line. Flags include:

* ```--control (-c)```: The method of control, from ['expert', 'manual', 'random', 'bc', 'bc-al']
* ```--randomize (-r)```: The degree of randomization, from ['m', 'g', ''] for randomized maze, goal, and none.
* ```--size (-s)```: If maze is randomized, the size of the generated maze.
* ```--maze (-m)```: If goal or none are randomized, the default maze configuration.
* ```--config (-y)```: The experiment configuration file, as a path to a yaml. Any additional arguments override yaml configs.
* ```--num_expert_samples (-N)```: For IRL/IL experiments, the number of trajectories sampled from the expert. This parameter helps with scripting experiments.


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
