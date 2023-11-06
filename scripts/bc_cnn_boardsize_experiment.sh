#!/bin/sh

# arr=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
arr=(6 8 10 12 16 20)

for i in ${arr[@]}
do
    python maze_runner.py -y configs/bc/bc_cnn_cfg.yaml -s $i
done
```

```bash
