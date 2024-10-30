"""
Ensemble of NeRF models that uses a set of M models to make predictions and 
compute uncertainty. 
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing
from typing import Optional, Tuple, Type, Literal, Mapping, Any, Dict

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import nn
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import copy

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.models.base_model import Model
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.cameras.cameras import Cameras

class VCURFPipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        config_path: Path,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        num_vcams: int = 10,
        sampling_radii_depth_ratio: float = 0.1,
        sampling_method: Literal["rgb", "depth"] = "rgb",
        grad_scaler: Optional[GradScaler] = None,
        # keep_origin_poses: bool = True,
    ):
        # config.datamanager.dataparser.keep_origin_poses = keep_origin_poses
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            grad_scaler=grad_scaler,
        )
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        _model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )

        self.model = _model
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

        self.num_vcams = num_vcams
        self.sampling_radii_depth_ratio = sampling_radii_depth_ratio
        self.sampling_method = sampling_method



    def get_ensemble_outputs_for_camera_ray_bundle(
            self,
            camera_ray_bundle,
            obb_box: Optional[OrientedBox] = None,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute the outputs for a given camera ray bundle and its nearby virtual cameras.
        
        Args:
            camera_ray_bundle: CameraRayBundle instance
            batch: Batch instance
        """
        outputs = self.model.get_outputs_for_camera(camera_ray_bundle, obb_box=obb_box)

        transform = camera_ray_bundle.metadata['transform']
        scale_factor = camera_ray_bundle.metadata['scale_factor']
        K_ = camera_ray_bundle.get_intrinsics_matrices().squeeze()  # (3,3)
        K = torch.eye(4).to(K_)
        K[:3, :3] = K_
        GetVCams = VirtualCameras(camera_ray_bundle)

        look_at_origin, D_center_origin, origin_rd_c2w = extract_scene_center_and_c2w(outputs['depth'].squeeze(), camera_ray_bundle)
        # D_median = outputs['depth'].clone().flatten().median(0).values
        # radiaus = self.sampling_radii_depth_ratio * D_median

        # look_at, D_center = extract_scene_center_and_c2w_v2(outputs['depth'].squeeze(), camera_ray_bundle)
        rd_c2w = torch.eye(4).to(origin_rd_c2w.device)
        rd_c2w[:3,:] = camera_ray_bundle.camera_to_worlds.squeeze(0)

        radiaus = self.sampling_radii_depth_ratio * D_center_origin
        Vcams = GetVCams.get_N_near_cam_by_look_at(self.num_vcams, look_at=look_at_origin, radiaus=radiaus)
        rd_depth = outputs['depth'].squeeze()
        rd_depth_origin = get_origin_depth(transform,scale_factor,K,rd_depth)
        # rd_depth = outputs['depth'].clone().permute(2,0,1)
        rd_depth_origin = rd_depth_origin.unsqueeze(0) #(1, h, w)

        # rd_depths = rd_depth.unsqueeze(0).repeat(self.num_vcams, 1, 1, 1)
        rd_depths_origin = rd_depth_origin.unsqueeze(0).repeat(self.num_vcams, 1, 1, 1)  # (N, h, w)
        pred_img = outputs['rgb'].clone().permute(2,0,1)
        pred_imgs = pred_img.unsqueeze(0).repeat(self.num_vcams, 1, 1, 1)

        vir_depths = []
        vir_pred_imgs = []
        rd2virs = []
        rd2rds = []
        #
        # transform_ = torch.tensor(camera_ray_bundle.metadata['transform']).to(rd_c2w.device)
        # transform = torch.eye(4).to(transform_)
        # transform[:3, :] = transform_

        backwarp = BackwardWarping(out_hw=(camera_ray_bundle.height.item(), camera_ray_bundle.width.item()),
                                   device=outputs['depth'].device, K=K)

        for vir_camera_ray_bundle in Vcams:
            vir_render_pkg = self.model.get_outputs_for_camera(vir_camera_ray_bundle, obb_box=obb_box)
            vir_depth = vir_render_pkg['depth'].squeeze()
            vir_depth_origin = (transform,scale_factor,K,vir_depth)

            vir_pred_img = vir_render_pkg['rgb'].permute(2,0,1)
            vir_c2w = torch.eye(4).to(rd_c2w)
            vir_c2w[:3,:] = vir_camera_ray_bundle.camera_to_worlds.squeeze()
            # vir_c2w = torch.inverse(transform) @ vir_c2w
            origin_vir_c2w = get_origin_pose(
                oriented_pose=vir_camera_ray_bundle.camera_to_worlds.squeeze(0),
                transform=vir_camera_ray_bundle.metadata['transform'],
                scale = vir_camera_ray_bundle.metadata['scale_factor']
            )

            rd2vir = torch.inverse(origin_rd_c2w) @ origin_vir_c2w
            # rd2vir = torch.inverse(vir_c2w) @ rd_c2w
            rd2rd = torch.inverse(rd_c2w) @ rd_c2w
            rd2virs.append(rd2vir)
            # vir_depths.append(vir_depth.unsqueeze(0))
            vir_depths.append(vir_depth_origin.unsqueeze(0))
            vir_pred_imgs.append(vir_pred_img)
        vir_depths = torch.stack(vir_depths)
        vir_pred_imgs = torch.stack(vir_pred_imgs)
        rd2virs = torch.stack(rd2virs)
        vir2rd_pred_imgs, vir2rd_depths, nv_mask = backwarp(img_src=vir_pred_imgs, depth_src=vir_depths,
                                                            depth_tgt=rd_depths_origin,
                                                            tgt2src_transform=rd2virs)
        ################################
        #  compute uncertainty by l2 diff
        ################################

        breakpoint()

        # depth uncertainty
        vir2rd_depth_sum = vir2rd_depths.sum(0)
        numels = float(self.num_vcams) - nv_mask.sum(0)
        vir2rd_depth = torch.zeros_like(rd_depth)
        vir2rd_depth[numels > 0] = vir2rd_depth_sum[numels > 0] / numels[numels > 0]
        depth_l2 = (rd_depth - vir2rd_depth) ** 2
        depth_l2[numels<1.] = 0.
        depth_std = torch.sqrt(depth_l2).squeeze(0)

        # rgb uncertainty
        vir2rd_pred_sum = vir2rd_pred_imgs.sum(0).mean(0, keepdim=True)
        rendering_ = pred_img.mean(0, keepdim=True)
        vir2rd_pred = torch.zeros_like(rendering_)
        vir2rd_pred[numels > 0] = vir2rd_pred_sum[numels > 0] / numels[numels > 0]
        rgb_l2 = (rendering_ - vir2rd_pred) ** 2
        rgb_l2[numels < 1.] = 0.
        rgb_std = torch.sqrt(rgb_l2).squeeze(0)

        if 'depth_std' in outputs.keys():
            outputs['depth_vc_std'] = depth_std.unsqueeze(-1)
        else:
            outputs['depth_std'] = depth_std.unsqueeze(-1)

        if 'rgb_std' in outputs.keys():
            outputs['rgb_vc_std'] = rgb_std.unsqueeze(-1)
        else:
            outputs['rgb_std'] = rgb_std.unsqueeze(-1)


        if self.sampling_method == 'depth':
            if 'rgb_vc_std' in outputs.keys():
                outputs['rgb_vc_std'] = outputs['depth_vc_std']
            else:
                outputs['rgb_std'] = outputs['depth_std']

        return outputs
    

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        is_inference: bool = False,
    ):
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        if is_inference:
            metrics_dict, images_dict = self.get_image_metrics_and_images_unc(
                outputs, batch, metrics_dict, images_dict
            )
        return metrics_dict, images_dict

class VirtualCameras:
    def __init__(self, Center_Camera):
        self.center_camera: Cameras = Center_Camera
        self.origin_camera_o,self.transform, self.origin_T = self.get_camera_center(self.center_camera)
        # self.sampling_center = self.get_camera_position(self.center_camera)
        self.data_device = self.origin_camera_o.device

    def get_camera_center(self,camera: Cameras):
        c2w_ = camera.camera_to_worlds.squeeze().clone() # (1,3,4)
        transform = torch.tensor(camera.metadata['transform']).to(c2w_.device)
        self.scale_factor = camera.metadata['scale_factor']
        self.transform = torch.eye(4).to(transform.device)
        self.transform[:3,:] = transform
        origin_c2w = get_origin_pose(c2w_, transform, self.scale_factor)
        origin_R = origin_c2w[:3, :3]
        origin_T = origin_c2w[:3, 3]
        camera_o = -origin_R.T @ origin_T
        return camera_o,self.transform, origin_T

    # def get_camera_position(self,camera: Cameras):
    #     c2w = camera.camera_to_worlds.squeeze().clone() # (1,3,4)
    #     translation = c2w[:3, 3]
    #     return translation

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
        points = torch.randn(N,3).to(O)
        points = 2*points-torch.ones_like(points)
        points = points / torch.norm(points, dim=1, keepdim=True)
        points = points * r + O

        return points

    def get_N_near_cam_by_look_at(self, N, look_at, radiaus=0.1):
        # sample new camera center
        new_translations = self.random_points_on_sphere(N,radiaus, self.origin_T)
        # new_translations = self.random_points_on_sphere(N,radiaus, self.sampling_center)
        Vcams = []
        look_at = look_at.to(self.origin_T.device)
        for new_t in new_translations:
            # create new camera pose by look at
            forward = look_at - new_t
            forward /= torch.linalg.norm(forward)
            forward = forward.to(new_t)
            world_up = torch.tensor(self.center_camera.metadata['world_up']).to(new_t.device)

            # world_up = torch.tensor([0., 0., 1.]).to(new_t.device)  # need to be careful for the openGL system!!!
            right = torch.cross(world_up, forward)
            right /= torch.linalg.norm(right)
            up = torch.cross(forward, right)
            new_R = torch.vstack([right, up, forward]).T
            # new_T = -new_R @ new_o
            new_T = new_t

            new_c2w = torch.eye(4).to(new_t.device)
            new_c2w[:3, :3] = new_R
            new_c2w[:3, 3] = new_T

            new_c2w = self.transform @ new_c2w
            new_c2w[:3, 3] *= self.scale_factor

            # new_c2w = self.center_camera.camera_to_worlds.squeeze().clone()
            # new_c2w[:3,3] = new_o

            new_camera = copy.deepcopy(self.center_camera)
            new_camera.camera_to_worlds = new_c2w[:3,:].unsqueeze(0)
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

def extract_scene_center_and_c2w(depth, camera):
    K_ = camera.get_intrinsics_matrices().squeeze() # 3x3
    K = torch.eye(4).to(K_)
    K[:3,:3] = K_
    inv_K = torch.inverse(K).unsqueeze(0)
    backproj_func = Backprojection(height=camera.image_height, width=camera.image_width)
    # C2W = torch.tensor(getView2World(view.R, view.T))
    oriented_c2w = camera.camera_to_worlds[0].clone().squeeze(0)  # (3,4)
    transform_ = camera.metadata['transform']
    transform = torch.eye(4).to(K_.device)
    transform[:3,4] = torch.tensor(transform_).to(K_.device)
    scale_factor = camera.metadata['scale_factor']
    origin_c2w = get_origin_pose(oriented_c2w, transform, scale_factor)
    origin_depth = get_origin_depth(transform,scale_factor,K,depth)

    depth_camera = depth.clone()
    depth_camera = depth_camera.unsqueeze(0).unsqueeze(0)
    # mask = (depth_v < depth_v.max()).squeeze(0)
    mask = (depth_camera > 0.).squeeze(0)
    point3d_camera = backproj_func(depth_camera.cpu(), inv_K.cpu(), img_like_out=True).squeeze(0) / scale_factor
    point3d_origin = torch.inverse(transform) @ point3d_camera.view(4, -1)

    point3d_world = origin_c2w.cpu() @ point3d_origin
    point3d_world = point3d_world.view(4, point3d_camera.shape[1], point3d_camera.shape[2])

    # expanded_mask = mask.expand_as(point3d_world)
    # selected = point3d_world.to(mask.device)[expanded_mask]
    # selected = selected.view(4, -1)
    # scene_center = selected.median(1).values[:3]
    # return scene_center, C2W_origin
    # return scene_center, C2W_nerf

    # if we assume the object is in the photo center
    h, w = point3d_world.shape[1], point3d_world.shape[2]
    center_patch = point3d_world[:, int(h // 2) - 8: int(h // 2) + 8, int(w // 2) - 8: int(w // 2) + 8]
    center_depth_path = origin_depth.squeeze()[int(h // 2) - 8: int(h // 2) + 8, int(w // 2) - 8: int(w // 2) + 8]

    scene_center = center_patch.reshape(4, -1).mean(-1)[:3]
    center_depth = center_depth_path.flatten().mean()

    return scene_center, center_depth, origin_c2w

def extract_scene_center_and_c2w_v2(depth, camera):
    # if we assume the object is in the photo center
    h, w = depth.shape[0], depth.shape[1]
    center_depth_path = depth.squeeze()[int(h // 2) - 8: int(h // 2) + 8, int(w // 2) - 8: int(w // 2) + 8]
    center_depth = center_depth_path.flatten().mean()

    c2w = camera.camera_to_worlds[0].clone().squeeze(0)  # (3,4)
    camera_position = c2w[:3, 3]
    forward_direction = -c2w[:3, 2]
    scene_center = camera_position + forward_direction * center_depth

    return scene_center, center_depth


def get_origin_pose(oriented_pose, transform, scale):
    C2W = torch.eye(4).to(oriented_pose.device)
    C2W[:3, :] = oriented_pose
    C2W[:3, 3] /= scale

    transform_matrix = torch.eye(4).to(oriented_pose.device)
    transform_matrix[:3, :] = torch.tensor(transform).to(C2W.device)

    origin_pose = torch.inverse(transform_matrix) @ C2W

    return origin_pose

def get_origin_depth(transform, scale, K, depth_oriented):
    transform = torch.tensor(transform).to(K.device)
    transform = torch.cat([transform,torch.tensor([[0,0,0,1]]).to(transform.device)],dim=0)

    height, width = depth_oriented.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # Shape: (H, W, 3)

    depth_oriented_flat = depth_oriented.view(-1)
    pixel_coords_flat = pixel_coords.view(-1, 3)

    points_oriented_camera = torch.linalg.inv(K) @ pixel_coords_flat.T * depth_oriented_flat
    points_oriented_camera /= scale

    points_oriented_camera_h = torch.cat([points_oriented_camera, torch.ones(1, points_oriented_camera.shape[1])],
                                           dim=0)
    points_original_camera_h = torch.linalg.inv(transform) @ points_oriented_camera_h
    points_original_camera = points_original_camera_h[:3]  # Drop the homogeneous row

    # points_image_h = (K @ points_original_camera).T
    # points_image = points_image_h[:, :2] / points_image_h[:, 2:3]

    depth_original_flat = points_original_camera[2]
    depth_original = depth_original_flat.view(height, width)
    return depth_original

# import torch
#
# # Assume we have:
# # depth_normalized: Depth map in normalized camera space (HxW)
# # K_normalized: Intrinsics of the normalized camera
# # K_original: Intrinsics of the original camera
# # T_normalized_to_original: Transformation from normalized camera to original camera pose (4x4 matrix)
#
# # Get the height and width of the depth map
# height, width = depth_normalized.shape
#
# # Create a meshgrid of pixel coordinates
# y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
# pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # Shape: (H, W, 3)
#
# # Back-project normalized depth map to 3D points in normalized camera space
# depth_normalized_flat = depth_normalized.view(-1)
# pixel_coords_flat = pixel_coords.view(-1, 3)
#
# # Convert to normalized camera 3D coordinates
# points_normalized_camera = torch.linalg.inv(K_normalized) @ pixel_coords_flat.T * depth_normalized_flat
#
# # Add a row of ones for homogeneous coordinates
# points_normalized_camera_h = torch.cat([points_normalized_camera, torch.ones(1, points_normalized_camera.shape[1])], dim=0)
#
# # Transform points to original camera pose
# points_original_camera_h = T_normalized_to_original @ points_normalized_camera_h
# points_original_camera = points_original_camera_h[:3]  # Drop the homogeneous row
#
# # Project points back to the original camera's image plane
# points_image_h = (K_original @ points_original_camera).T
#
# # Normalize by z to get pixel coordinates in the original image space
# points_image = points_image_h[:, :2] / points_image_h[:, 2:3]
#
# # Recover depth (z-coordinate) in the original camera space
# depth_original_flat = points_original_camera[2]
#
# # Reshape to HxW
# depth_original = depth_original_flat.view(height, width)
#
# print("Recovered Depth Map for Original Pose:", depth_original)
