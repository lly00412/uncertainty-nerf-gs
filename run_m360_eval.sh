#!/bin/bash
CUDA_VISIBLE_DEVICES=1

# Train nerfacto for VS-NeRF

#ns-eval-unc vcurf-config --load-config ./output/m360/kitchen_v30/kitchen/active-nerfacto/main/config.yml \
#            --sampling-method rgb    \
#            --output-path ./output/m360/kitchen_v30/kitchen/vcurf/active-nerfacto/main/rgb_l2/metrics.json   \
#            --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/    \
#            --render-output-path ./output/m360/kitchen_v30/kitchen/vcurf/active-nerfacto/main/rgb_l2/plots \
#            --save-rendered-images --unc-max 0.3




ns-eval-unc vcurf-config --load-config ./output/m360/kitchen_v30/kitchen/nerfacto-mcdropout/main/config.yml \
            --sampling-method depth   \
            --num_vcams 6 \
            --output-path ./output/m360/kitchen_v30/kitchen/vcurf/nerfacto-mcdropout/main/depth_l2/metrics.json     \
            --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/       \
            --render-output-path ./output/m360/kitchen_v30/kitchen/vcurf/nerfacto-mcdropout/main/depth_l2/plots     \
            --save-rendered-images --unc-max 0.3


ns-eval-unc vcurf-config --load-config ./output/m360/garden_v20/garden/active-nerfacto/main/config.yml \
            --sampling-method rgb   \
            --num_vcams 6 \
            --output-path ./output/m360/garden_v20/garden/vcurf/active-nerfacto/main/rgb_l2/metrics.json     \
            --dataset-path  /mnt/Data2/nerf_datasets/m360/garden/       \
            --render-output-path ./output/m360/garden_v20/garden/vcurf/active-nerfacto/main/rgb_l2/plots     \
            --save-rendered-images --unc-max 0.3