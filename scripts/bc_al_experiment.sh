#!/bin/sh

# arr=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
arr=(1 5 10 15 20 30 40 50)

# Just use big size

# arr=(30 50)
for i in ${arr[@]}
do
    python maze_runner.py -y configs/bc/bc_al_cfg.yaml -N $i
done