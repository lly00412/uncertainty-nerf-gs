#!/bin/bash

# Train nerfacto for VS-NeRF
ns-train nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ \
          --output-dir output/m360/kitchen_v30 \
          --machine.seed 24030 \
          --experiment-name kitchen/24030 \
          --method-name nerfacto --timestamp main \
          --pipeline.model.camera-optimizer.mode off \
          --viewer.quit-on-train-completion True sparse-mipnerf360v2 \
          --num_train_images 30 --downscale-factor 4

ns-train nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ \
          --output-dir output/m360/kitchen_v30 \
          --machine.seed 11430 \
          --experiment-name kitchen/11430 \
          --method-name nerfacto --timestamp main \
          --pipeline.model.camera-optimizer.mode off \
          --viewer.quit-on-train-completion True sparse-mipnerf360v2 \
          --num_train_images 30 --downscale-factor 4

ns-train nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ \
          --output-dir output/m360/kitchen_v30 \
          --machine.seed 6759 \
          --experiment-name kitchen/6759 \
          --method-name nerfacto --timestamp main \
          --pipeline.model.camera-optimizer.mode off \
          --viewer.quit-on-train-completion True sparse-mipnerf360v2 \
          --num_train_images 30 --downscale-factor 4

python nerfuncertainty/scripts/eval_uncertainty.py ensemble-config \
        --load-config ./output/m360/kitchen_v30/kitchen/11430/nerfacto/main/config.yml ./output/m360/kitchen_v30/kitchen/24030/nerfacto/main/config.yml ./output/m360/kitchen_v30/kitchen/6759/nerfacto/main/config.yml  \
        --output-path ./output/m360/kitchen_v30/kitchen/ensembling/main/metrics.json \
        --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/  \
        --render-output-path ./output/m360/kitchen_v30/kitchen/ensembling/main/plots --save-rendered-images --unc-max 0.3