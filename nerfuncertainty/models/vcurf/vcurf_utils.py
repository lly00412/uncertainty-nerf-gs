# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import copy

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import Cameras


from nerfuncertainty.models.ensemble.ensemble_pipeline import EnsemblePipeline, EnsemblePipelineSplatfacto


class VirtualCameras:
    def __init__(self, Center_Camera):
        self.center_camera: Cameras = Center_Camera
        self.sampling_center = self.get_camera_center(self.center_camera)
        self.data_device = self.sampling_center.device
    def get_camera_center(self,camera: Cameras):
        c2w = camera.camera_to_worlds # (1,3,4)
        camera_o = c2w[0,:3,3].clone()
        return camera_o
    def random_points_on_sphere(self, N, r, O):
        """
        Generate N random points on a sphere of radius r centered at O.

        Args:
        - N: number of points
        - r: radius of the sphere
        - O: center of the sphere as a tensor of shape (3,), i.e., O = torch.tensor([x, y, z])

        Returns:
        - points: a tensor of shape (N, 3) representing the N random points on the sphere
        """
        points = torch.rand(N,3).to(O)
        points = 2*points-torch.ones_like(points)
        points = points / torch.norm(points, dim=1, keepdim=True)
        points = points * r
        points = points + O
        return points

    def get_N_near_cam_by_look_at(self, N, look_at, radiaus=0.1):
        # sample new camera center
        new_centers = self.random_points_on_sphere(N,radiaus, self.sampling_center)
        Vcams = []
        for new_o in new_centers:
            # create new camera pose by look at
            forward = look_at - new_o
            forward /= torch.linalg.norm(forward)
            forward = forward.to(new_o)
            world_up = torch.tensor([0, 1, 0]).to(new_o)  # need to be careful for the openGL system!!!
            right = torch.cross(world_up, forward)
            right /= torch.linalg.norm(right)
            up = torch.cross(forward, right)

            new_c2w = torch.eye(4).to(new_o)
            new_c2w[:3, :3] = torch.vstack([right, up, forward]).T
            new_c2w[:3, 3] = new_o
            new_c2w = new_c2w[:3,:]

            new_camera = copy.deepcopy(self.center_camera)
            new_camera.camera_to_worlds = new_c2w.unsqueeze(0)
            Vcams.append(new_camera)

        return Vcams

class Projection(nn.Module):
    """Layer which projects 3D points into a camera view
    """
    def __init__(self, height, width, eps=1e-7):
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points3d, K, normalized=True):
        """
        Args:
            points3d (torch.tensor, [N,4,(HxW)]: 3D points in homogeneous coordinates
            K (torch.tensor, [torch.tensor, (N,4,4)]: camera intrinsics
            normalized (bool):
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            xy (torch.tensor, [N,H,W,2]): pixel coordinates
        """
        # projection
        points2d = torch.matmul(K[:, :3, :], points3d)

        # convert from homogeneous coordinates
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + self.eps)
        xy = xy.view(points3d.shape[0], 2, self.height, self.width)
        xy = xy.permute(0, 2, 3, 1)

        # normalization
        if normalized:
            xy[..., 0] /= self.width - 1
            xy[..., 1] /= self.height - 1
            xy = (xy - 0.5) * 2
        return xy

class Transformation3D(nn.Module):
    """Layer which transform 3D points
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self,
                points: torch.Tensor,
                T: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
            T (torch.Tensor, [N,4,4]): transformation matrice
        Returns:
            transformed_points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points

class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics

    Attributes
        xy (torch.tensor, [N,3,HxW]: homogeneous pixel coordinates on regular grid
    """

    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        self.xy = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
            , 0)
        self.xy = torch.cat([self.xy, self.ones], 1)
        self.xy = nn.Parameter(self.xy, requires_grad=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (torch.tensor, [N,1,H,W]: depth map
            inv_K (torch.tensor, [N,4,4]): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is [N,4,H,W]; else [N,4,(HxW)]
        Returns:
            points (torch.tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1)
        ones = self.ones.repeat(depth.shape[0], 1, 1)

        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points

class BackwardWarping(nn.Module):

    def __init__(self,
                 out_hw: Tuple[int,int],
                 device: torch.device,
                 K:torch.Tensor) -> None:
        super(BackwardWarping,self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height,width).to(device)
        self.projection = Projection(height,width).to(device)
        self.transform3d = Transformation3D().to(device)

        H,W = height,width
        self.rgb = torch.zeros(H,W,3).view(-1,3).to(device)
        self.depth = torch.zeros(H, W, 1).view(-1, 1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K).to(device)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0) # 1,4,4
    def forward(self,
                img_src: torch.Tensor,
                depth_src: torch.Tensor,
                depth_tgt: torch.Tensor,
                tgt2src_transform: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = depth_tgt.shape

        # reproject
        pts3d_tgt = self.backproj(depth_tgt,self.inv_K)
        pts3d_src = self.transform3d(pts3d_tgt,tgt2src_transform)
        src_grid = self.projection(pts3d_src,self.K,normalized=True)
        transformed_distance = pts3d_src[:, 2:3].view(b,1,h,w)

        img_tgt = F.grid_sample(img_src, src_grid, mode = 'bilinear', padding_mode = 'zeros')
        depth_src2tgt = F.grid_sample(depth_src, src_grid, mode='bilinear', padding_mode='zeros')

        # rm invalid depth
        valid_depth_mask = (transformed_distance < 1e6) & (depth_src2tgt > 0)

        # rm invalid coords
        vaild_coord_mask = (src_grid[...,0]> -1) & (src_grid[...,0] < 1) & (src_grid[...,1]> -1) & (src_grid[...,1] < 1)
        vaild_coord_mask = vaild_coord_mask.unsqueeze(1)

        valid_mask = valid_depth_mask & vaild_coord_mask
        invaild_mask = ~valid_mask

        return img_tgt.float(), depth_src2tgt.float(), invaild_mask.float()

def extract_scene_center(depth, camera):
    K_ = camera.get_intrinsics_matrices # 3x3
    K = torch.eye(4).to(K_)
    K[:3,:3] = K_
    inv_K = torch.inverse(K).unsqueeze(0)
    backproj_func = Backprojection(height=camera.image_height, width=camera.image_width)
    depth_v = depth.clone()
    depth_v = depth_v.unsqueeze(0).unsqueeze(0)
    # mask = (depth_v < depth_v.max()).squeeze(0)
    mask = (depth_v > 0.).squeeze(0)
    point3d_camera = backproj_func(depth_v.cpu(), inv_K.cpu(), img_like_out=True).squeeze(0)
    # C2W = torch.tensor(getView2World(view.R, view.T))
    C2W_ = camera.camera_to_worlds[0].clone().squeeze(0) # (3,4)
    C2W = torch.eye(4).to(C2W_)

    point3d_world = C2W.cpu() @ point3d_camera.view(4, -1)
    point3d_world = point3d_world.view(4, point3d_camera.shape[1], point3d_camera.shape[2])
    expanded_mask = mask.expand_as(point3d_world)
    selected = point3d_world.to(mask.device)[expanded_mask]
    selected = selected.view(4, -1)
    scene_center = selected.median(1).values[:3]

    return scene_center



def eval_load_ensemble_checkpoints(
    config: TrainerConfig, pipeline: Pipeline, config_paths: Tuple[Path, ...]
) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print(f"Loading latest checkpoint from {config.load_dir}")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                f"No checkpoint directory found at {config.load_dir}, ",
                justify="center",
            )
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(
            int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir)
        )[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")

    for model_idx in range(1, len(pipeline.models)):
        config_i = yaml.load(config_paths[model_idx].read_text(), Loader=yaml.Loader)
        assert isinstance(config_i, TrainerConfig)
        load_dir = config_paths[model_idx].parent / "nerfstudio_models"
        config_i.load_dir = load_dir
        if config_i.load_step is None:
            CONSOLE.print(f"Loading latest checkpoint from {load_dir}")
            # NOTE: this is specific to the checkpoint name format
            if not os.path.exists(load_dir):
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"No checkpoint directory found at {load_dir}, ",
                    justify="center",
                )
                CONSOLE.print(
                    "Please make sure the checkpoint exists, they should be generated periodically during training",
                    justify="center",
                )
                sys.exit(1)
            load_step = sorted(
                int(x[x.find("-") + 1 : x.find(".")])
                for x in os.listdir(config_i.load_dir)
            )[-1]
        else:
            load_step = config_i.load_step
        load_path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        pipeline.load_pipeline(
            loaded_state["pipeline"], loaded_state["step"], model_idx=model_idx
        )
        CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    pipeline.models.to(pipeline.device)
    return load_path, load_step


def ensemble_eval_setup(
    config_paths: List[Path],
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_paths[0].read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[
        config.method_name
    ].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()
    if isinstance(config.pipeline.datamanager, VanillaDataManagerConfig):
        config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add methods applicable for ensemble
    if config.method_name == "nerfacto":
        config.pipeline._target = EnsemblePipeline
    elif config.method_name == "active-nerfacto":
        config.pipeline._target = EnsemblePipeline
    elif config.method_name == "splatfacto":
        config.pipeline._target = EnsemblePipelineSplatfacto
    elif config.method_name == "active-splatfacto":
        config.pipeline._target = EnsemblePipelineSplatfacto

    pipeline = config.pipeline.setup(
        device=device, test_mode=test_mode, config_paths=config_paths
    )
    assert isinstance(pipeline, EnsemblePipeline) or isinstance(pipeline, EnsemblePipelineSplatfacto)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_ensemble_checkpoints(
        config, pipeline, config_paths=config_paths
    )

    return config, pipeline, checkpoint_path, step
