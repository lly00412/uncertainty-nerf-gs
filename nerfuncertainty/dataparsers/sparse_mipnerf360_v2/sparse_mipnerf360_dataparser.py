
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Literal, Optional

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio

@dataclass
class SparseMipNerf360DataParserConfigv2(NerfstudioDataParserConfig):
    """MipNerf360 dataset parser config"""

    _target: Type = field(default_factory=lambda: SparseMipNerf360v2)
    """target class to instantiate"""
    num_train_images: int = 10
    """How many images use in training"""
    seed_random_split: int = 2024 #Literal['seed1', 'seed2', 'seed3'] = 'seed1'
    """Which random split to use"""

@dataclass
class SparseMipNerf360v2(Nerfstudio):
    """Nerfstudio DatasetParser"""

    config: SparseMipNerf360DataParserConfigv2
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                # assert self.config.proportion_train_images > 0.0 or self.config.proportion_train_images <= 1.0 #, "proportion of trainig images must in [0.0, 1.0]"
                # randomly sample indices
                # rng = torch.Generator().manual_seed(self.config.seed_random_split)
                # perm = torch.randperm(len(i_train), generator=rng)
                # num_images = int(np.round(len(i_train)*self.config.proportion_train_images))
                # inds = perm[:num_images]
                # do uniform sampling
                inds = np.linspace(0,len(i_train)-1,self.config.num_train_images,dtype=int)
                i_train_ = list(np.array(i_train)[inds])
                indices = i_train_
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")



        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = data_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import create_ply_from_colmap

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    ply_filename = "sparse_pc.ply"
                    create_ply_from_colmap(
                        filename=ply_filename,
                        recon_dir=colmap_path,
                        output_dir=self.config.data,
                        applied_transform=applied_transform,
                    )
                    ply_file_path = data_dir / ply_filename
                    transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(self.config.data / "transforms.json", "w", encoding="utf-8") as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )
        return dataparser_outputs