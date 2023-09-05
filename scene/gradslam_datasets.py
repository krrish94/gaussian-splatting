"""
PyTorch dataset classes for datasets in the NICE-SLAM format.
Large chunks of code stolen and adapted from:
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py

Support for Replica (sequences from the iMAP paper), TUM RGB-D, NICE-SLAM Apartment.
TODO: Add Azure Kinect dataset support
"""

import abc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from natsort import natsorted


def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True,  # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"] if "crop_size" in config_dict["camera_params"] else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError("Mismatch between number of color images and number of embedding files.")
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()

        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )


class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: Dict,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[Union[Path, str]] = "embeddings",
        embedding_dim: Optional[int] = 512,
        embedding_file_extension: Optional[str] = "pt",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))
        if self.pose_path == 0:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = self.pose_path[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        lines = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        _posearr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            _npvec = np.asarray([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
            _posearr.append(_npvec)
        _posearr = np.stack(_posearr)

        for pose_line_idx in range(0, _posearr.shape[0], 3):
            _curpose = np.zeros((4, 4))
            _curpose[3, 3] = 3
            _curpose[0] = _posearr[pose_line_idx]
            _curpose[1] = _posearr[pose_line_idx + 1]
            _curpose[2] = _posearr[pose_line_idx + 2]
            poses.append(torch.from_numpy(_curpose).float())

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ScannetDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Ai2thorDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            if self.embedding_dir == "embed_semseg":
                # embed_semseg is stored as uint16 pngs
                embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png"))
            else:
                embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        if self.embedding_dir == "embed_semseg":
            embedding = imageio.imread(embedding_file_path)  # (H, W)
            embedding = cv2.resize(
                embedding, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST
            )
            embedding = torch.from_numpy(embedding).long()  # (H, W)
            embedding = F.one_hot(embedding, num_classes=self.embedding_dim)  # (H, W, C)
            embedding = embedding.half()  # (H, W, C)
            embedding = embedding.permute(2, 0, 1)  # (C, H, W)
            embedding = embedding.unsqueeze(0)  # (1, C, H, W)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class AzureKinectDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None

        # # check if a file named 'poses_global_dvo.txt' exists in the basedir / sequence folder
        # if os.path.isfile(os.path.join(basedir, sequence, "poses_global_dvo.txt")):
        #     self.pose_path = os.path.join(basedir, sequence, "poses_global_dvo.txt")

        if "odomfile" in kwargs.keys():
            self.pose_path = os.path.join(self.input_folder, kwargs["odomfile"])
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.pose_path is None:
            print("WARNING: Dataset does not contain poses. Returning identity transform.")
            return [torch.eye(4).float() for _ in range(self.num_imgs)]
        else:
            # Determine whether the posefile ends in ".log"
            # a .log file has the following format for each frame
            # frame_idx frame_idx+1
            # row 1 of 4x4 transform
            # row 2 of 4x4 transform
            # row 3 of 4x4 transform
            # row 4 of 4x4 transform
            # [repeat for all frames]
            #
            # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
            # 16 entries of 4x4 transform
            # [repeat for all frames]
            if self.pose_path.endswith(".log"):
                # print("Loading poses from .log format")
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                if len(lines) % 5 != 0:
                    raise ValueError(
                        "Incorrect file format for .log odom file " "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5
                for i in range(0, num_lines):
                    _curpose = []
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))
                    _curpose = np.array(_curpose).reshape(4, 4)
                    poses.append(torch.from_numpy(_curpose))
            else:
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if len(line.split()) == 0:
                        continue
                    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                    poses.append(torch.from_numpy(c2w))
            return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding  # .permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class RealsenseDataset(GradSLAMDataset):
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")))
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, "depth", "*.png")))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Record3DDataset(GradSLAMDataset):
    """
    Dataset class to read in saved files from the structure created by our
    `save_record3d_stream.py` script
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        load_seg: Optional[bool] = False,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "poses")
        self.load_seg = load_seg
        if load_seg:
            self.seg_path = os.path.join(self.input_folder, "seg")
            self.load_seg = True
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.png")))
        if len(color_paths) == 0:
            # Search for .jpg extensions too
            color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")))
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, "depth", "*.png")))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        seg_paths = None
        if self.load_seg:
            self.seg_paths = natsorted(glob.glob(os.path.join(self.input_folder, "seg", "*.png")))
            self.seg_paths = self.seg_paths[self.start : self.end : stride]
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        if self.load_seg:
            seg_path = self.seg_paths[index]
            seg = np.asarray(imageio.imread(seg_path), dtype=np.int64)
            seg = torch.from_numpy(seg)
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                seg.to(self.device).type(self.dtype),
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )


def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings


def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":
    import open3d as o3d
    from gradslam.slam.pointfusion import PointFusion
    from gradslam.structures.pointclouds import Pointclouds
    from gradslam.structures.rgbdimages import RGBDImages
    from tqdm import trange

    # # # ICL
    # cfg = load_dataset_config("/home/krishna/code/gradslam-foundation/examples/dataconfigs/icl.yaml")
    # gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/icl"
    # dataset = ICLDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/data/icl",
    #     sequence="living_room_traj1_frei_png",
    #     start=0,
    #     end=-1,
    #     stride=30,
    #     desired_height=240,
    #     desired_width=320,
    # )
    # Replica dataset
    cfg = load_dataset_config("/home/krishna/code/gradslam-foundation/examples/dataconfigs/replica/replica.yaml")
    gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-graphs/liam-lab-scan11"
    dataset = ReplicaDataset(
        config_dict=cfg,
        basedir="/home/krishna/data/nice-slam-data/Replica",
        sequence="room0",
        start=0,
        end=-1,
        stride=10,
        desired_height=240,
        desired_width=320,
    )
    # # # Azure Kinect (Ali UdeM lab scan 11)
    # cfg = load_dataset_config("/home/krishna/code/gradslam-foundation/examples/dataconfigs/azure/ali-udem.yaml")
    # gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-graphs/liam-lab-scan11"
    # dataset_image_width = cfg["camera_params"]["image_width"]
    # dataset_image_height = cfg["camera_params"]["image_height"]
    # dataset = AzureKinectDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/data/concept-graphs/",
    #     sequence="liam-lab-scan11",
    #     start=0,
    #     end=-1,
    #     stride=100,
    #     desired_height=240,
    #     desired_width=320,
    #     odomfile="odomfile_rtabmap.txt",
    # )
    # # # Record3D capture (Krishna BCS room)
    # cfg = load_dataset_config("/home/krishna/data/record3d/krishna-bcs-room/dataconfig.yaml")
    # gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-fields/krishna-bcs-room"
    # dataset_image_width = cfg["camera_params"]["image_width"]
    # dataset_image_height = cfg["camera_params"]["image_height"]
    # dataset = Record3DDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/data/record3d/",
    #     sequence="krishna-bcs-room",
    #     start=0,
    #     end=-1,
    #     stride=20,
    #     desired_height=120,
    #     desired_width=160,
    #     # odomfile="odomfile_rtabmap.txt",
    # )
    # # # Record3D capture (Krishna BCS room -- for concept fields -- with seg)
    # cfg = load_dataset_config(
    #     "/home/krishna/code/gaussian-splatting/data/krishna-bcs-office-longer-seq/dataconfig.yaml"
    # )
    # gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-fields/krishna-bcs-office-longer-seq"
    # ply_path = "data/concept-fields/krishna-bcs-office-longer-seq/splat_input_point_cloud.ply"
    # dataset_image_width = cfg["camera_params"]["image_width"]
    # dataset_image_height = cfg["camera_params"]["image_height"]
    # dataset = Record3DDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/code/gaussian-splatting/data/",
    #     sequence="krishna-bcs-office-longer-seq",
    #     start=0,
    #     end=-1,
    #     stride=10,
    #     desired_height=160,
    #     desired_width=120,
    #     # odomfile="odomfile_rtabmap.txt",
    # )

    device = torch.device("cuda:0")

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        has_embeddings=False,  # KM
    )

    slam = PointFusion(odom="gt", dsratio=1, device=device, use_embeddings=False)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(
        device=device,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    frame_cur, frame_prev = None, None

    print("Running PointFusion (incremental mode)...")

    for idx in trange(len(dataset)):
        _color, _depth, intrinsics, _pose, *_ = dataset[idx]

        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0).float(),
            _pose.unsqueeze(0).unsqueeze(0).float(),
        )
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)

    # pcd = pointclouds.open3d(0)
    # o3d.visualization.draw_geometries([pcd])

    # Save pointcloud
    pointclouds.save_to_h5(gradslam_pcd_out_path, include_embeddings=False)
