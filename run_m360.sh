#!/bin/bash

## Train mcdropout
#ns-train nerfacto-mcdropout --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_v30 --experiment-name kitchen --method-name nerfacto-mcdropout --timestamp main --pipeline.model.camera-optimizer.mode off --pipeline.model.density-dropout-layers False --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 30 --downscale-factor 4
#python nerfuncertainty/scripts/eval_uncertainty.py mc-dropout-config --load-config output/m360/kitchen_v30/kitchen/nerfacto-mcdropout/main/config.yml  --output-path output/m360/kitchen_v30/kitchen/nerfacto-mcdropout/main/metrics.json --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/ --render-output-path output/m360/kitchen_v30/kitchen/nerfacto-mcdropout/main/plots --save-rendered-images --unc-max 0.3 --mc_samples 10
#
## Train activenerf
#ns-train active-nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_v30 --experiment-name kitchen --method-name active-nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 30 --downscale-factor 4
#python nerfuncertainty/scripts/eval_uncertainty.py active-nerfacto-config --load-config output/m360/kitchen_0.1/kitchen/active-nerfacto/main/config.yml  --output-path output/m360/kitchen_v30/kitchen/active-nerfacto/main/metrics.json --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/ --render-output-path output/m360/kitchen_v30/kitchen/active-nerfacto/main/plots --save-rendered-images --unc-max 0.3
#
## Train nerfacto for VS-NeRF
#ns-train nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_v30 --experiment-name kitchen --method-name nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 30 --downscale-factor 4
#
#
#ns-eval-unc vcurf-config
#
#ns-eval-unc vcurf-config --load-config ./output/m360/kitchen_v30/kitchen/nerfacto/main/config.yml \
#        --output-path ./output/m360/kitchen_v30/kitchen/vcurf/main/rgb_l2/metrics.json \
#        --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/  \
#        --render-output-path ./output/m360/kitchen_v30/kitchen/vcurf/main/rgb_l2/plots --save-rendered-images --unc-max 0.3
#
#ns-eval-unc vcurf-config --load-config ./output/m360/kitchen_v30/kitchen/nerfacto/main/config.yml \
#        --sampling-method depth \
#        --output-path ./output/m360/kitchen_v30/kitchen/vcurf/main/depth_l2/metrics.json \
#        --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/  \
#        --render-output-path ./output/m360/kitchen_v30/kitchen/vcurf/main/depth_l2/plots --save-rendered-images --unc-max 0.3
#
#
#!/bin/bash
CUDA_VISIBLE_DEVICES=1

# Train nerfacto for VS-NeRF

#ns-train nerfacto-mcdropout --data /mnt/Data2/nerf_datasets/m360/garden/ \
#        --pipeline.model.density_dropout_layers False \
#        --output-dir output/m360/garden_v20 --experiment-name garden \
#        --method-name nerfacto-mcdropout --timestamp main \
#        --pipeline.model.camera-optimizer.mode off \
#        --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 30 --downscale-factor 4

ns-eval-unc vcurf-config --load-config ./output/m360/garden_v20/garden/nerfacto-mcdropout/main/config.yml \
        --sampling-method depth \
        --output-path ./output/m360/garden_v20/garden/vcurf/nerfacto-mcdropout/main/depth_l2/metrics.json \
        --dataset-path  /mnt/Data2/nerf_datasets/m360/garden/  \
        --render-output-path ./output/m360/garden_v20/garden/vcurf/nerfacto-mcdropout/main/depth_l2/plots --save-rendered-images --unc-max 0.3