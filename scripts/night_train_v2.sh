#!/bin/bash
set -e
while true; do
  if python -m src.training.train_v2 --envs 1 --steps 1000000 --ckpt_dir models/checkpoints \ 
     && python -m src.inference.export --sb3 --ckpt models/checkpoints/best_sb3.zip --out ssl_policy.ts; then
    mkdir -p models/exported
    cp ssl_policy.ts models/exported/ssl_policy.ts
    sleep 5
  else
    echo "[night_train] restart after error"
    sleep 5
  fi
done
