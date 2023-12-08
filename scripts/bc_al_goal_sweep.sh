#!/bin/sh

echo "RUNNGIN ED"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_ed_cfg.yaml
echo "RUNNGIN GM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_gm_cfg.yaml
echo "RUNNGIN GAM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_gam_cfg.yaml
echo "RUNNGIN RANDOM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_random_cfg.yaml

echo "RUNNGIN ED"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_ed_cfg_3.yaml
echo "RUNNGIN GM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_gm_cfg_3.yaml
echo "RUNNGIN GAM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_gam_cfg_3.yaml
echo "RUNNGIN RANDOM"
python maze_runner.py --config ./configs/bc/bc-al/goal_randomized/bc_al_random_cfg_3.yaml