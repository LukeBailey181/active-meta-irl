#!/bin/sh

# arr=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
# arr=(250 500 750 1000 1250 1500 1750 2000)
arr=(2500 3000 3500 4000 5000)

for i in ${arr[@]}
do
    python maze_runner.py -y configs/bc/bc_cnn_cfg.yaml -N $i
done

