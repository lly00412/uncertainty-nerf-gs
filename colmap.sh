#!/bin/bash

ROOT_DIR=/mnt/Data2/nerf_datasets/m360/
SCENES=(garden bicycle bonsai counter flowers room stump treehill)

for scene in "${SCENES[@]}"; do
    cd $ROOT_DIR/$scene
    cp -r images images_copy
    ns-process-data images --data $ROOT_DIR/$scene/images_copy --output-dir $ROOT_DIR/$scene --verbose
done