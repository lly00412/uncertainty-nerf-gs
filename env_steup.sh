#!/bin/bash
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# build comap
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86


ns-train active-nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_0.1 --experiment-name kitchen --method-name active-nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 10 --downscale-factor 4

python nerfuncertainty/scripts/eval_uncertainty.py active-nerfacto-config --load-config output/m360/kitchen_0.1/kitchen/active-nerfacto/main/config.yml  --output-path output/m360/kitchen_0.1/kitchen/active-nerfacto/main/metrics.json --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/ --render-output-path output/m360/kitchen_0.1/kitchen/active-nerfacto/main/plots --save-rendered-images --unc-max 0.3


ns-train nerfacto-mcdropout --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_v80 --experiment-name kitchen --method-name nerfacto-mcdropout --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 80 --downscale-factor 4

ns-train nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_0.1 --experiment-name kitchen --method-name nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 10 --downscale-factor 4


python nerfuncertainty/scripts/eval_uncertainty.py mc-dropout-config --load-config output/m360/kitchen_0.1/kitchen/nerfacto-mcdropout/main/config.yml  --output-path output/m360/kitchen_0.1/kitchen/nerfacto-mcdropout/main/metrics.json --dataset-path  /mnt/Data2/nerf_datasets/m360/kitchen/ --render-output-path output/m360/kitchen_0.1/kitchen/nerfacto-mcdropout/main/plots --save-rendered-images --unc-max 0.3 --mc_samples 10

ns-train nerfacto-laplace --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_0.1 --experiment-name kitchen --method-name nerfacto-laplace --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360v2 --num_train_images 10 --downscale-factor 4