#!/bin/sh

echo "Running ED eval"
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_ed_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/ed ./logs/evals/bc-al-maze-new/ed/ed.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_ed_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/ed_2 ./logs/evals/bc-al-maze-new/ed/ed.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_ed_cfg_3.yaml  ./logs/bc-al/bc-al-maze-size-10/ed_3 ./logs/evals/bc-al-maze-new/ed/ed.csv

echo "Running GAM"
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gam_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/gam ./logs/evals/bc-al-maze-new/gam/gam.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gam_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/gam_2 ./logs/evals/bc-al-maze-new/gam/gam.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gam_cfg_3.yaml  ./logs/bc-al/bc-al-maze-size-10/gam_3 ./logs/evals/bc-al-maze-new/gam/gam.csv

echo "Running GM"
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gm_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/gm ./logs/evals/bc-al-maze-new/gm/gm.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gm_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/gm_2 ./logs/evals/bc-al-maze-new/gm/gm.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_gm_cfg_3.yaml  ./logs/bc-al/bc-al-maze-size-10/gm_3 ./logs/evals/bc-al-maze-new/gm/gm.csv

echo "Random"
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_random_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/random ./logs/evals/bc-al-maze-new/random/random.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_random_cfg.yaml  ./logs/bc-al/bc-al-maze-size-10/random_2 ./logs/evals/bc-al-maze-new/random/random.csv
./scripts/evaluate_models.sh ./configs/bc/bc-al/maze_randomized/bc_cnn_al_random_cfg_3.yaml  ./logs/bc-al/bc-al-maze-size-10/random_3 ./logs/evals/bc-al-maze-new/random/random.csv