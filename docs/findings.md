# SC3K Architecture & Training Pipeline Analysis

This document compiles the structural, technical, and algorithmic findings for the **SC3K** self-supervised 3D keypoint estimation architecture (ICCV 2023).

---

## 1. SC3K Training Pipeline Workflow

```
            +---------------------------------+
            |   Canonical 3D Point Cloud      |
            +---------------------------------+
                             |
              [Apply 2 Random Camera Poses]
                             |
             +---------------+---------------+
             |                               |
             v                               v
     +---------------+               +---------------+
     | PCD 1 (Pose 1)|               | PCD 2 (Pose 2)|
     +---------------+               +---------------+
             |                               |
             v                               v
     +---------------+               +---------------+
     | PointNet Feat |               | PointNet Feat |
     +---------------+               +---------------+
             |                               |
             v                               v
     +---------------+               +---------------+
     | Residual Blocks               | Residual Blocks
     | + Softmax     |               | + Softmax     |
     +---------------+               +---------------+
             |                               |
             v                               v
     +---------------+               +---------------+
     |   Keypoints   |               |   Keypoints   |
     |   KP1 (K x 3) |               |   KP2 (K x 3) |
     +---------------+               +---------------+
             |                               |
             +---------------+---------------+
                             |
                             v
          +-------------------------------------+
          |    Self-Supervised Auxiliary Losses |
          |  - Shape / Volume / Consistency     |
          |  - Separation / Overlap / Pose      |
          +-------------------------------------+
                             |
                             v
          +-------------------------------------+
          |       Adam Weight Optimization      |
          +-------------------------------------+
```

* **Data Prep**: Extracts raw point clouds -> rotates them using two camera matrices -> feeds paired views (`pcd1`, `pcd2`) into a shared encoder network.
* **Forward Pass**: PointNet computes point features -> MLPs & Residual blocks squeeze features to $K$ dimensions -> Spatial Softmax assigns point-wise keypoint probabilities -> Coordinate weighted average outputs predicted coordinates.
* **Backward Pass**: Multi-task self-supervised loss functions calculate discrepancy metrics -> Adam optimizer backpropagates gradients and updates shared weights.

---

## 2. Configuration Analysis (`config/config.yaml`)

* **`split`**: Runs training/validation modes if set to `train`. Runs evaluation if set to `test`.
* **`task`**:
  * `generic`: Training & evaluation under random rotations and noise (enables paired consistency loss).
  * `canonical`: Evaluation on standard, unaligned objects without transformations.
* **`key_points`**: Number of keypoints ($K$) to predict. Standard is `10` for ShapeNet classes.
* **`batch_size`**: Number of point clouds processed in parallel (set to `26`).
* **`max_epoch`**:
  * **Current value (`5`)**: Insufficient for real convergence. Used strictly for fast pipeline debugging.
  * **Recommended value (`200`)**: Target parameter from the ICCV 2023 paper to reach optimal results.
* **`overlap_threshold`**: Safe distance barrier (`0.05`) below which keypoints are penalized for collision.
* **Loss Weights (`parameters`)**:
  * `separation` (`1`): Repels keypoints from clustering.
  * `overlap` (`1`): Strict penalty if keypoints breach the overlap threshold.
  * `shape` (`6`): Pulls keypoints onto the physical surface of the point cloud.
  * `consist` (`1`): Demands keypoints remain matching across rotations.
  * `volume` (`1`): Forces keypoints to span the full bounding volume of the object.
  * `pose` (`0.07`): Enforces structural constellation stability against ground-truth rotation.
* **Augmentation**:
  * `normalize_pc` (`True`): Scales point cloud to unit sphere $[-1, 1]$ (essential for network scale-invariance).
  * `gaussian_noise` (`False`): Set to `True` with `lamda`/`lamda2` to inject noise and replicate paper's noise robustness.
  * `down_sample` (`False`): Triggers Farthest Point Sampling (FPS) to decimate points to a fixed count (`2048`).

---

## 3. Epochs, Batches, and Execution Phases

### Basic Concepts
* **Batch**: Small group of samples (e.g. 26 point clouds) computed simultaneously to perform one parameter update.
* **Epoch**: One complete pass through the entire dataset.
  * $\text{Iterations per Epoch} = \text{Total Samples} / \text{Batch Size}$

### Execution Cycle

```
+--------------------------------------------------------------------------+
|                              TRAINING PHASE                              |
|  - Enable model.train()                                                  |
|  - Load batch -> forward pass -> calculate loss                          |
|  - Zero gradients -> loss.backward() -> optimizer.step()                 |
+--------------------------------------------------------------------------+
                                     |
                          [At Epoch Completion]
                                     |
                                     v
+--------------------------------------------------------------------------+
|                             VALIDATION PHASE                             |
|  - Enable model.eval() & with torch.no_grad()                            |
|  - Load validation batch -> forward pass -> compute loss                 |
|  - Average validation losses across validation set                       |
|  - IF val_loss < best_loss THEN save weights to Best_class_10kp.pth     |
+--------------------------------------------------------------------------+
```

---

## 4. Keypoint Losses Deep Dive

### Overlap Loss vs. Separation Loss
```
  [Overlap Loss: Hard Barrier]             [Separation Loss: Continuous Repulsion]
    (No effect if dist > 0.05)                (Always active: pushes points apart)
         
          O           O                            <-- O           O -->
           \   <--   /
            O-------O
```
* **Overlap Loss**:
  * **Function**: Strict collision detector.
  * **Mechanism**: Counts pairs closer than `overlap_threshold` (0.05) and adds a direct penalty.
  * **Gradient**: Step-like activation; inactive ($0$ loss) once points are sufficiently separated.
* **Separation Loss**:
  * **Function**: Continuous spreading force.
  * **Mechanism**: Computes inverse of the average nearest-neighbor distance: $1 / \text{mean}(d_{\text{closest}})$.
  * **Gradient**: Continuous; constantly repels keypoints to maximize coverage across the object's geometry.

### Consistency Loss vs. Pose Loss
```
  [Consistency Loss (Spatial Frame)]             [Pose Loss (Rotation Space)]
    (Apply gt pose, match absolute pts)            (Estimate pose from KPs, match R)

          View 1           View 2                       KP1              KP2
         +------+         +------+                        \              /
         |  o   |  <====  |  o   |                         \-[Align SVD]-/
         |      |  (Pose) |      |                                |
         +------+         +------+                                v
                                                             Estimated R_est
                                                                  vs.
                                                             Ground-Truth R_gt
```
* **Consistency Loss**:
  * **Function**: Semantic point consistency across views.
  * **Mechanism**: Projects view 2 keypoints (`kp2`) into view 1's frame using ground-truth rotation, then computes spatial point-to-point MSE against `kp1`.
* **Pose Loss**:
  * **Function**: Rigid constellation alignment.
  * **Mechanism**: Uses Orthogonal Procrustes alignment (SVD) on predicted points (`kp1`, `kp2`) to estimate an orientation matrix, then calculates angular error compared to ground-truth rotation.

---

## 5. Multi-Class Constraints

* **Category-Specific Design**:
  * The network is intended to learn semantic parts (e.g. wingtips for airplanes, backrest for chairs).
  * A single model cannot generalize keypoints across classes because different categories have distinct physical structures and topologies.
* **Dataset Logic**:
  * `generic_data_loader` filters annotation json files to load one single `class_name` at a time.
* **Training Requirement**:
  * **You must train a separate set of weights for each object class.**
