import json
import os
import sys
from pathlib import Path

import cv2
import deeplake
import numpy as np
from PIL import Image
from tqdm import tqdm, trange


def main():
    ds = deeplake.load("data/krishna-bcs-office-longer-seq-ordered")

    outdir = "data/krishna-bcs-office-longer-seq"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Create directories named "rgb", "depth", "poses", "seg"
    outdir_rgb = os.path.join(outdir, "rgb")
    outdir_depth = os.path.join(outdir, "depth")
    outdir_poses = os.path.join(outdir, "poses")
    outdir_seg = os.path.join(outdir, "seg")
    Path(outdir_rgb).mkdir(parents=True, exist_ok=True)
    Path(outdir_depth).mkdir(parents=True, exist_ok=True)
    Path(outdir_poses).mkdir(parents=True, exist_ok=True)
    Path(outdir_seg).mkdir(parents=True, exist_ok=True)

    # keys of interest: "image", "depth", "metadata", "pose", "sam/sequence_segmentation/instance_mask"
    # print(ds["image"])
    print("Saving images...")
    for idx, img in tqdm(enumerate(ds["image"]), total=len(ds["image"])):
        img = img.numpy()
        # Write out image as cv2 jpg output
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(outdir_rgb, f"{idx:06d}.jpg"), img)

    print("Saving depth...")
    for idx, depth in tqdm(enumerate(ds["depth"]), total=len(ds["depth"])):
        depth = depth.numpy()[..., 0]
        depth = depth.astype(np.uint16)
        depth = Image.fromarray(depth)
        depth.save(os.path.join(outdir_depth, f"{idx:06d}.png"))

    print("Saving segmentation...")
    for idx, seg in tqdm(
        enumerate(ds["sam/sequence_segmentation/instance_mask"]),
        total=len(ds["sam/sequence_segmentation/instance_mask"]),
    ):
        seg = seg.numpy()
        seg = seg.astype(np.uint16)
        seg = Image.fromarray(seg)
        seg.save(os.path.join(outdir_seg, f"{idx:06d}.png"))

    print("Saving poses...")
    for idx, pose in tqdm(enumerate(ds["pose"]), total=len(ds["pose"])):
        pose = pose.numpy()
        pose = pose.astype(np.float32)
        # Save pose
        np.save(os.path.join(outdir_poses, f"{idx:06d}.npy"), pose)


if __name__ == "__main__":
    main()
