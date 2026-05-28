import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Generate unique random poses for SC3K training classes."
)
parser.add_argument(
    "--class_id",
    type=str,
    default=None,
    help="Specific ShapeNet class synset ID (e.g., '02691156' for Airplanes). If omitted, processes all classes found in dataset/pcds/"
)
args = parser.parse_args()

# Determine class IDs to process
if args.class_id:
    class_ids = [args.class_id]
else:
    pcd_root = "dataset/pcds"
    if not os.path.exists(pcd_root):
        print(f"Error: Directory '{pcd_root}' not found. Please check your dataset symlinks.")
        exit(1)
    
    class_ids = [
        d for d in os.listdir(pcd_root) 
        if os.path.isdir(os.path.join(pcd_root, d))
    ]
    
    if not class_ids:
        print(f"Error: No class directories found inside '{pcd_root}'.")
        exit(1)

# A standard camera intrinsic matrix (Ignored by SC3K's math, but required by the dataloader)
dummy_camera_mat = np.array(
    [[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32
)

# A dummy translation vector (Ignored by SC3K)
dummy_translation = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

print(f"Found {len(class_ids)} class(es) to process: {', '.join(class_ids)}")

for class_id in class_ids:
    pcd_dir = f"dataset/pcds/{class_id}"
    pose_dir = f"dataset/poses/{class_id}"
    
    if not os.path.exists(pcd_dir):
        print(f"\n[Warning] PCD directory for class {class_id} does not exist at '{pcd_dir}'. Skipping.")
        continue

    # Automatically create the target pose directory if it doesn't exist
    os.makedirs(pose_dir, exist_ok=True)
    
    pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
    if not pcd_files:
        print(f"\n[Warning] No .pcd files found in '{pcd_dir}'. Skipping.")
        continue
        
    print(f"\nProcessing class {class_id}...")
    print(f"Generating unique random poses for {len(pcd_files)} models...")
    
    created_count = 0
    for pcd_path in pcd_files:
        obj_id = os.path.basename(pcd_path).replace(".pcd", "")
        target_npz_path = os.path.join(pose_dir, f"{obj_id}.npz")
        
        # Generate 24 uniquely random 3D rotations for this specific model
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
        
    print(f"Successfully generated {created_count} unique .npz pose files inside '{pose_dir}'!")

print("\nPose generation complete!")
