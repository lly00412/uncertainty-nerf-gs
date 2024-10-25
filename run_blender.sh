#!/bin/bash
CUDA_VISIBLE_DEVICES=1

# Train nerfacto for VS-NeRF

ns-train nerfacto --data /mnt/Data2/nerf_datasets/nerf_synthetic/ship/ \
        --output-dir output/nerf_synthetic/ship_v80 --experiment-name ship \
        --method-name nerfacto --timestamp main \
        --pipeline.model.camera-optimizer.mode off \
        --viewer.quit-on-train-completion True sparse-blender --num_images 80 \
        --seed_random_split seed4

ns-train nerfacto-mcdropout --data /mnt/Data2/nerf_datasets/nerf_synthetic/ship/ \
        --pipeline.model.density_dropout_layers False \
        --pipeline.model.dropout_rate 0.1 \
        --output-dir output/nerf_synthetic/ship_v80 --experiment-name ship \
        --method-name nerfacto-mcdropout --timestamp main \
        --pipeline.model.camera-optimizer.mode off \
        --viewer.quit-on-train-completion True sparse-blender --num_images 80 \
        --seed_random_split seed4

ns-eval-unc vcurf-config --load-config ./output/nerf_synthetic/ship_v80/ship/nerfacto-mcdropout/main/config.yml \
        --sampling-method depth \
        --output-path ./output/nerf_synthetic/ship_v80/ship/vcurf/nerfacto-mcdropout/main/depth_l2/metrics.json \
        --dataset-path  /mnt/Data2/nerf_datasets/nerf_synthetic/ship/  \
        --render-output-path ./output/nerf_synthetic/ship_v80/ship/vcurf/nerfacto-mcdropout/main/depth_l2/plots --save-rendered-images --unc-max 0.3

#ns-eval-unc vcurf-config --load-config ./output/m360/kitchen_v30/kitchen/nerfacto-mcdropout/main/config.yml \
#            --sampling-method depth   \
#            --num_vcams 8 \
#            --output-path ./output/m360/kitchen_v30/kitchen/vcurf/nerfacto-mcdropout/main/depth_l2/metrics.json     \
#            --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/       \
#            --render-output-path ./output/m360/kitchen_v30/kitchen/vcurf/nerfacto-mcdropout/main/depth_l2/plots     \
#            --save-rendered-images --unc-max 0.3
