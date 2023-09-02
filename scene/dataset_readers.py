#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    seg: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir)
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def read_gradslam_dataset():
    print("In read_gradslam_dataset")
    from gradslam.structures import Pointclouds

    from .gradslam_datasets import (
        AzureKinectDataset,
        Record3DDataset,
        ReplicaDataset,
        load_dataset_config,
    )

    # # # Replica scene
    # cfg = load_dataset_config("/home/krishna/code/gradslam-foundation/examples/dataconfigs/replica/replica.yaml")
    # ply_path = "data/replica/room0/splat_input_point_cloud.ply"
    # gradslam_pcd_out_path = "data/replica/room0"
    # dataset_image_width = cfg["camera_params"]["image_width"]
    # dataset_image_height = cfg["camera_params"]["image_height"]
    # dataset = ReplicaDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/data/nice-slam-data/Replica",
    #     sequence="room0",
    #     start=0,
    #     end=-1,
    #     stride=20,
    #     desired_height=dataset_image_height,
    #     desired_width=dataset_image_width,
    # )
    # # # Azure Kinect capture (Liam lab scan 11)
    # ply_path = "data/concept-graphs/liam-lab-scan11/splat_input_point_cloud.ply"
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
    #     desired_height=dataset_image_height,
    #     desired_width=dataset_image_width,
    #     odomfile="odomfile_rtabmap.txt",
    # )
    # # # Record3D capture (Krishna BCS room)
    # cfg = load_dataset_config("/home/krishna/data/record3d/krishna-bcs-room/dataconfig.yaml")
    # gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-fields/krishna-bcs-room"
    # ply_path = "data/concept-fields/krishna-bcs-room/splat_input_point_cloud.ply"
    # dataset_image_width = cfg["camera_params"]["image_width"]
    # dataset_image_height = cfg["camera_params"]["image_height"]
    # dataset = Record3DDataset(
    #     config_dict=cfg,
    #     basedir="/home/krishna/data/record3d/",
    #     sequence="krishna-bcs-room",
    #     start=0,
    #     end=-1,
    #     stride=10,
    #     desired_height=dataset_image_height,
    #     desired_width=dataset_image_width,
    #     # odomfile="odomfile_rtabmap.txt",
    # )
    # # Record3D capture (Krishna BCS room -- for concept fields -- with seg)
    load_seg = True
    cfg = load_dataset_config(
        "/home/krishna/code/gaussian-splatting/data/krishna-bcs-office-longer-seq/dataconfig.yaml"
    )
    gradslam_pcd_out_path = "/home/krishna/code/gaussian-splatting/data/concept-fields/krishna-bcs-office-longer-seq"
    ply_path = "data/concept-fields/krishna-bcs-office-longer-seq/splat_input_point_cloud.ply"
    dataset_image_width = cfg["camera_params"]["image_width"]
    dataset_image_height = cfg["camera_params"]["image_height"]
    dataset = Record3DDataset(
        config_dict=cfg,
        basedir="/home/krishna/code/gaussian-splatting/data/",
        sequence="krishna-bcs-office-longer-seq",
        start=0,
        end=-1,
        stride=10,
        desired_height=dataset_image_height,
        desired_width=dataset_image_width,
        load_seg=load_seg,
        # odomfile="odomfile_rtabmap.txt",
    )

    # Replicate the readColmapCameras steps here, accounting for the gradslam dataset format
    cam_infos_unsorted = []

    # _, _, intrinsics, *_ = dataset[0]
    fovy = focal2fov(cfg["camera_params"]["fy"], dataset_image_height)
    fovx = focal2fov(cfg["camera_params"]["fx"], dataset_image_width)

    for idx in range(len(dataset)):
        _, _, _, pose, seg = dataset[idx]
        image_path = dataset.color_paths[idx]
        # print(image_path)
        image = Image.open(image_path)
        import torch

        pose = torch.linalg.inv(pose)
        R = pose[:3, :3].detach().cpu().numpy()
        R = R.T
        t = pose[:3, 3].detach().cpu().numpy()
        image_name = os.path.basename(image_path).split(".")[0]
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=t,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=dataset_image_width,
            height=dataset_image_height,
            fx=cfg["camera_params"]["fx"],
            fy=cfg["camera_params"]["fy"],
            cx=cfg["camera_params"]["cx"],
            cy=cfg["camera_params"]["cy"],
            seg=seg.detach().cpu().numpy(),
        )
        cam_infos_unsorted.append(cam_info)

    # cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    cam_infos = cam_infos_unsorted
    train_cam_infos = cam_infos
    test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Read the gradslam pointcloud
    pcd = Pointclouds.load_pointcloud_from_h5(gradslam_pcd_out_path)
    _points = pcd.points_padded.detach().cpu().numpy()[0][::50]
    _colors = pcd.colors_padded.detach().cpu().numpy()[0][::50]
    storePly(ply_path, _points, _colors)

    scene_info = SceneInfo(
        point_cloud=fetchPly(ply_path),
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        # nerf_normalization={"translate": np.array([0, 0, 0]), "radius": 10.0},
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "gradslam": read_gradslam_dataset,
}
