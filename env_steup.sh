#!/bin/bash
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# build comap
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86


ns-train active-nerfacto --data /mnt/Data2/nerf_datasets/m360/kitchen/ --output-dir output/m360/kitchen_0.1 --experiment-name kitchen --method-name active-nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360 --proportion_train_images 0.1 --downscale-factor 4