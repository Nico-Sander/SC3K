# Dataset Setup & Linking Guide

This guide outlines the steps to download, link, and pre-process the **KeypointNet** dataset for training the **SC3K** self-supervised 3D keypoint estimation pipeline.

---

## Prerequisites & Directory Structure

To keep the `SC3K` repository lightweight and clean, the actual dataset files are hosted in your local `KeypointNet` repository and linked to `SC3K` using symbolic links (symlinks). 

Your workspace should ideally look like this:
```text
workspace/
├── KeypointNet/          # Cloned KeypointNet repository
│   ├── annotations/      # Extracted annotation files
│   ├── pcds/             # Extracted point clouds (.pcd)
│   ├── splits/           # Train/val/test splits (.json)
│   └── poses/            # Generated poses (.npz)
└── SC3K/                 # SC3K project root
    └── dataset/          # Symlinked folders pointing to KeypointNet
```

---

## Setup Steps

### 1. Clone the KeypointNet Repository
Clone the original KeypointNet repository to your workspace:
```bash
git clone https://github.com/nileshkulkarni/KeypointNet.git
```

### 2. Download and Place Dataset Files
1. Download the dataset archives (annotations, point clouds, splits) from the official KeypointNet Google Drive release.
2. Extract the contents and place the directories directly into your local `KeypointNet` repository:
   - `KeypointNet/annotations/`
   - `KeypointNet/pcds/`
   - `KeypointNet/splits/`
3. Create an empty directory inside `KeypointNet` to serve as the root folder for target camera poses:
   ```bash
   mkdir -p KeypointNet/poses
   ```
*(Note: Subdirectories for specific ShapeNet class IDs will be created automatically during pose generation).*

### 3. Link the Dataset to SC3K
Create a `dataset` folder in your `SC3K` root directory and set up symbolic links pointing to the directories inside `KeypointNet`. 

From your `SC3K` root directory, run:
```bash
# Create the dataset container directory
mkdir -p dataset

# Create symlinks (replace /path/to/ with your actual workspace path)
ln -s /path/to/KeypointNet/annotations dataset/annotations
ln -s /path/to/KeypointNet/pcds dataset/pcds
ln -s /path/to/KeypointNet/splits dataset/splits
ln -s /path/to/KeypointNet/poses dataset/poses
```

### 4. Pre-generate Training Poses
SC3K requires paired views with predefined camera transformations during self-supervised training. Run the pose generation script from the `SC3K` root directory to generate 24 unique random SO(3) rotations per model.

The pose generator script supports **automatic class detection** and **CLI customization**:

#### Option A: Auto-process All Available Classes (Recommended)
This command scans `dataset/pcds/` and automatically generates camera rotation transformations for *all* found ShapeNet categories (it will also automatically create the target directories):
```bash
python3 generate_poses.py
```

#### Option B: Target a Specific ShapeNet Class
To only generate poses for a specific category, pass its ShapeNet Synset ID using the `--class_id` option:
```bash
# Target airplanes (synset ID: 02691156)
python3 generate_poses.py --class_id 02691156

# Target chairs (synset ID: 03001627)
python3 generate_poses.py --class_id 03001627
```

---

## Verification
You can verify your links and files by running:
```bash
ls -l dataset
```
You should see active symbolic links pointing to your `KeypointNet` repository paths, and `dataset/poses/` should contain subfolders for each processed class ID containing unique `.npz` files for every object model.
