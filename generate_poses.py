import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

pcd_dir = "dataset/pcds/02691156"
pose_dir = "dataset/poses/02691156"

# Get all .pcd files in the dataset
pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
print(f"Generating unique random poses for {len(pcd_files)} models...")

# A standard camera intrinsic matrix (Ignored by SC3K's math, but required by the dataloader)
dummy_camera_mat = np.array(
    [[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32
)

# A dummy translation vector (Ignored by SC3K)
dummy_translation = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

created_count = 0
for pcd_path in pcd_files:
    obj_id = os.path.basename(pcd_path).replace(".pcd", "")
    target_npz_path = os.path.join(pose_dir, f"{obj_id}.npz")

    # Generate 24 uniquely random 3D rotations for this specific airplane
    # This exactly mimics the random SO(3) rotations used in ONet/PointView-GCN
    random_rotations = R.random(24).as_matrix().astype(np.float32)

    pose_dict = {}
    for i in range(24):
        # Combine the 3x3 random rotation with the 3x1 translation to make a 3x4 world_mat
        world_mat = np.hstack((random_rotations[i], dummy_translation))

        pose_dict[f"world_mat_{i}"] = world_mat
        pose_dict[f"camera_mat_{i}"] = dummy_camera_mat

    # Overwrite any existing files so we get fresh, unique rotations
    np.savez(target_npz_path, **pose_dict)
    created_count += 1

print(f"Successfully generated {created_count} unique .npz pose files!")
