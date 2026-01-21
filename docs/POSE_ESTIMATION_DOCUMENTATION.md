# Pose Estimation Documentation

## Overview

The OCL-ReID system uses **FastPose** (from the AlphaPose framework) for 2D human pose estimation. This is a **top-down** approach where persons are first detected, then pose is estimated for each detected person.

---

## Table of Contents

1. [Algorithm Overview](#1-algorithm-overview)
2. [Architecture](#2-architecture)
3. [Features Extracted](#3-features-extracted)
4. [Data Flow](#4-data-flow)
5. [Keypoint Definitions](#5-keypoint-definitions)
6. [Feature Processing](#6-feature-processing)
7. [Usage in Pipeline](#7-usage-in-pipeline)
8. [File References](#8-file-references)

---

## 1. Algorithm Overview

| Component | Description |
|-----------|-------------|
| **Name** | FastPose (from AlphaPose) |
| **Type** | Top-down single-person pose estimation (SPPE) |
| **Backbone** | SE-ResNet (ResNet50 or ResNet101 with Squeeze-and-Excitation blocks) |
| **Upsampling** | DUC (Dense Upsampling Convolution) + PixelShuffle |
| **Output** | 17 keypoint heatmaps (COCO format), reduced to 13 (eyes/ears removed) |
| **Input Size** | 256×192 (ResNet50) or 320×256 (ResNet101) |

### Key Features

- **Top-down approach**: First detect persons (YOLOX), then estimate pose for each person crop
- **SE-ResNet backbone**: Uses Squeeze-and-Excitation blocks for channel-wise attention
- **DUC upsampling**: More accurate than bilinear/deconvolution for high-resolution heatmaps
- **Pretrained weights**: `fast_res50_256x192.pth` or `fast_res101_320x256.pth`

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FastPose Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input Image Patch (256×192 or 320×256)                            │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────┐                                           │
│   │   SE-ResNet50/101   │  ← Backbone with Squeeze-and-Excitation   │
│   │   (Feature Extract) │    Channels: 3 → 2048                     │
│   │   Output: 8×6×2048  │                                           │
│   └──────────┬──────────┘                                           │
│              │                                                       │
│              ▼                                                       │
│   ┌─────────────────────┐                                           │
│   │  PixelShuffle (2×)  │  ← Upsampling: 8×6 → 16×12                │
│   │  Channels: 2048→512 │                                           │
│   └──────────┬──────────┘                                           │
│              │                                                       │
│              ▼                                                       │
│   ┌─────────────────────┐                                           │
│   │   DUC Layer 1 (2×)  │  ← Dense Upsampling Convolution           │
│   │   16×12 → 32×24     │    Channels: 512 → 256                    │
│   └──────────┬──────────┘                                           │
│              │                                                       │
│              ▼                                                       │
│   ┌─────────────────────┐                                           │
│   │   DUC Layer 2 (2×)  │  ← Dense Upsampling Convolution           │
│   │   32×24 → 64×48     │    Channels: 256 → 128                    │
│   └──────────┬──────────┘                                           │
│              │                                                       │
│              ▼                                                       │
│   ┌─────────────────────┐                                           │
│   │   Conv2d (3×3)      │  ← Output layer                           │
│   │   Channels: 128→17  │    17 keypoint heatmaps                   │
│   └──────────┬──────────┘                                           │
│              │                                                       │
│              ▼                                                       │
│   17 Keypoint Heatmaps (64×48)                                      │
│   (reduced to 13 by removing eyes and ears)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Code (FastPose.py)

```python
class FastPose(nn.Module):
    DIM = 128

    def __init__(self, backbone='resnet101', num_join=17):
        super(FastPose, self).__init__()
        
        self.preact = SEResnet(backbone)      # SE-ResNet backbone
        self.suffle1 = nn.PixelShuffle(2)     # 2× upsampling
        self.duc1 = DUC(512, 1024, upscale_factor=2)   # DUC layer 1
        self.duc2 = DUC(256, 512, upscale_factor=2)    # DUC layer 2
        self.conv_out = nn.Conv2d(self.DIM, num_join, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)      # Backbone
        out = self.suffle1(out)   # PixelShuffle
        out = self.duc1(out)      # DUC 1
        out = self.duc2(out)      # DUC 2
        out = self.conv_out(out)  # Output heatmaps
        return out
```

---

## 3. Features Extracted

### 3.1 Raw Output from Model

| Feature | Shape | Description |
|---------|-------|-------------|
| `pose_hm` | `(N, 17, H, W)` | Heatmaps for 17 keypoints (reduced to 13) |
| `maxval` | `(N, 13, 1)` | **Confidence scores** for each keypoint |
| `xy_hm` | `(N, 13, 2)` | Keypoint locations in heatmap coordinates |
| `xy_img` | `(N, 13, 2)` | **Keypoint locations** transformed to image coordinates |

### 3.2 After NMS Processing (Final Output)

The `pose_nms()` function returns a list of dictionaries, one per person:

```python
{
    'bbox': Tensor(4),           # [x1, y1, x2, y2] - bounding box
    'bbox_score': Tensor(1),     # Detection confidence score
    'bbox_id': Tensor(1),        # Track ID
    'keypoints': Tensor(13, 2),  # (x, y) coordinates for 13 joints
    'kp_score': Tensor(13, 1),   # Confidence score for each keypoint
    'proposal_score': float      # Combined score
}
```

**Proposal Score Calculation:**
```python
proposal_score = mean(kp_score) + bbox_score + 1.25 * max(kp_score)
```

### 3.3 Stored in Pipeline (`track_kpts`)

```python
# In part_rpf.py
track_kpts.append(torch.cat((ps['keypoints'], ps['kp_score']), axis=1).tolist())
# Shape: (13, 3) per person → [x, y, confidence] for each joint
```

### 3.4 Processed in Tracklet (`joints_feature`)

The keypoints are further processed into a **joints_feature** vector:

```python
# In tracklet.py
def get_joints_feature(self, img_metas, kpts):
    """
    Input:  kpts (14, 3) - [x, y, confidence] for 14 joints (13 + Neck)
    Output: joints_feature (28,) - Softmax-weighted normalized coordinates
    """
    # 1. Normalize coordinates to [0, 1]
    scaled_kpts[:, 0] = kpts[:, 0] / image_width
    scaled_kpts[:, 1] = kpts[:, 1] / image_height
    
    # 2. Mask low-confidence keypoints (conf < 0.3)
    mask = kpts[:, 2] > 0.3
    scaled_kpts = scaled_kpts * mask
    
    # 3. Flatten to 1D: (14, 2) → (28,)
    scaled_kpts = scaled_kpts.flatten()
    
    # 4. Apply softmax with temperature=0.1
    joints_feature = softmax(scaled_kpts / 0.1)
    
    return joints_feature  # Shape: (28,)
```

### 3.5 Summary of All Saved Features

| Feature Name | Location | Shape | Purpose |
|--------------|----------|-------|---------|
| `track_kpts` | `raw_result` | `List[N × (13, 3)]` | Raw keypoints + confidence per person |
| `kpts` | `Tracklet` | `(14, 3)` | 14 joints (13 + computed Neck) |
| `joints_feature` | `Tracklet` | `(28,)` | Softmax-normalized joint positions for classification |
| `bbox_feature` | `Tracklet` | `(2,)` | Normalized bbox height/width |

---

## 4. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pose Estimation Data Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐            │
│  │ Input Image  │────▶│ Track BBoxes  │────▶│   Crop Persons   │            │
│  │  (H, W, 3)   │     │  (N, 5)       │     │  (N, 3, 256,192) │            │
│  └──────────────┘     └───────────────┘     └────────┬─────────┘            │
│                                                      │                       │
│                                                      ▼                       │
│                                          ┌──────────────────────┐            │
│                                          │    FastPose Model    │            │
│                                          │   (SE-ResNet + DUC)  │            │
│                                          └────────┬─────────────┘            │
│                                                   │                          │
│                                                   ▼                          │
│                                          ┌──────────────────────┐            │
│                                          │  Heatmaps (N,17,H,W) │            │
│                                          │  Remove eyes/ears    │            │
│                                          │  → (N, 13, H, W)     │            │
│                                          └────────┬─────────────┘            │
│                                                   │                          │
│                                                   ▼                          │
│                                          ┌──────────────────────┐            │
│                                          │   getPrediction()    │            │
│                                          │  • Find max in heatmap│           │
│                                          │  • Transform to image │           │
│                                          └────────┬─────────────┘            │
│                                                   │                          │
│                                                   ▼                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        pose_nms() Output                           │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │  Per Person Dictionary:                                      │   │     │
│  │  │  • bbox: [x1, y1, x2, y2]                                   │   │     │
│  │  │  • bbox_score: float                                         │   │     │
│  │  │  • bbox_id: int (track ID)                                   │   │     │
│  │  │  • keypoints: (13, 2) - [x, y] for each joint               │   │     │
│  │  │  • kp_score: (13, 1) - confidence for each joint            │   │     │
│  │  │  • proposal_score: float                                     │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                   │                          │
│                    ┌──────────────────────────────┼──────────────────┐       │
│                    │                              │                  │       │
│                    ▼                              ▼                  ▼       │
│           ┌────────────────┐           ┌────────────────┐   ┌──────────────┐│
│           │  track_kpts    │           │  Orientation   │   │   Tracklet   ││
│           │  (13, 3) each  │           │  Estimation    │   │  • kpts(14,3)││
│           │  [x, y, conf]  │           │  (HOE Network) │   │  • joints_   ││
│           └────────────────┘           └────────────────┘   │    feature   ││
│                                                             │    (28,)     ││
│                                                             └──────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Keypoint Definitions

### 5.1 Original COCO 17 Keypoints

```
Index  Joint Name      Status
─────  ──────────────  ──────────
0      Nose            ✓ Kept
1      Left Eye        ✗ Removed
2      Right Eye       ✗ Removed
3      Left Ear        ✗ Removed
4      Right Ear       ✗ Removed
5      Left Shoulder   ✓ Kept
6      Right Shoulder  ✓ Kept
7      Left Elbow      ✓ Kept
8      Right Elbow     ✓ Kept
9      Left Wrist      ✓ Kept
10     Right Wrist     ✓ Kept
11     Left Hip        ✓ Kept
12     Right Hip       ✓ Kept
13     Left Knee       ✓ Kept
14     Right Knee      ✓ Kept
15     Left Ankle      ✓ Kept
16     Right Ankle     ✓ Kept
```

### 5.2 Final 14 Keypoints Used (After Processing)

```
Index  Joint Name       Source
─────  ──────────────   ────────────────────────────
0      Nose             Original keypoint 0
1      Left Shoulder    Original keypoint 5
2      Right Shoulder   Original keypoint 6
3      Left Elbow       Original keypoint 7
4      Right Elbow      Original keypoint 8
5      Left Wrist       Original keypoint 9
6      Right Wrist      Original keypoint 10
7      Left Hip         Original keypoint 11
8      Right Hip        Original keypoint 12
9      Left Knee        Original keypoint 13
10     Right Knee       Original keypoint 14
11     Left Ankle       Original keypoint 15
12     Right Ankle      Original keypoint 16
13     Neck             COMPUTED: (LShoulder + RShoulder) / 2
```

### 5.3 Keypoint Skeleton Visualization

```
                    [0] Nose
                       │
                   [13] Neck (computed)
                    /     \
        [1] L.Shoulder   [2] R.Shoulder
              │                 │
        [3] L.Elbow       [4] R.Elbow
              │                 │
        [5] L.Wrist       [6] R.Wrist



        [7] L.Hip ─────────── [8] R.Hip
              │                 │
        [9] L.Knee       [10] R.Knee
              │                 │
       [11] L.Ankle      [12] R.Ankle
```

---

## 6. Feature Processing

### 6.1 Heatmap to Keypoint Conversion

```python
def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    """
    Get keypoint location from heatmaps
    
    Args:
        hms: Heatmaps (N, 13, H, W)
        pt1, pt2: Crop coordinates for inverse transform
        inpH, inpW: Input image size
        resH, resW: Heatmap resolution
    
    Returns:
        preds: Keypoint positions in heatmap coords (N, 13, 2)
        preds_tf: Keypoint positions in image coords (N, 13, 2)
        maxval: Confidence scores (N, 13, 1)
    """
    # 1. Find argmax in each heatmap
    maxval, idx = torch.max(hms.view(N, 13, -1), dim=2)
    
    # 2. Convert flat index to (x, y)
    preds_x = (idx - 1) % hms.size(3)
    preds_y = (idx - 1) // hms.size(3)
    
    # 3. Mask invalid predictions (maxval <= 0)
    preds *= (maxval > 0)
    
    # 4. Transform from heatmap coords to image coords
    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)
    
    return preds, preds_tf, maxval
```

### 6.2 Joints Feature Computation

```python
def get_joints_feature(self, img_metas, kpts):
    """
    Convert raw keypoints to normalized feature vector
    
    Args:
        img_metas: Image metadata (contains img_shape)
        kpts: Raw keypoints (14, 3) - [x, y, confidence]
    
    Returns:
        joints_feature: Tensor (28,) - Softmax-normalized positions
    """
    h, w, _ = img_metas[0]['img_shape']
    
    # Step 1: Normalize coordinates to [0, 1]
    scaled_kpts = np.zeros((14, 2))
    scaled_kpts[:, 0] = kpts[:, 0] / w  # x / width
    scaled_kpts[:, 1] = kpts[:, 1] / h  # y / height
    
    # Step 2: Apply confidence mask (threshold = 0.3)
    mask = kpts[:, 2] > 0.3
    scaled_kpts = scaled_kpts * np.expand_dims(mask, axis=1)
    
    # Step 3: Flatten to 1D vector
    scaled_kpts = scaled_kpts.flatten()  # (14, 2) → (28,)
    
    # Step 4: Apply softmax with temperature
    temperature = 0.1
    joints_feature = softmax(scaled_kpts / temperature)
    
    return torch.Tensor(joints_feature)  # Shape: (28,)
```

### 6.3 Neck Keypoint Computation

```python
# In tracklet.py
# Add neck as average of left and right shoulders
self.kpts = np.concatenate((
    self.kpts,  # Original 13 keypoints
    np.expand_dims(
        (np.array(self.kpts)[1, :] + np.array(self.kpts)[2, :]) / 2,  # (LShoulder + RShoulder) / 2
        axis=0
    )
), axis=0)
# Result: (14, 3) keypoints
```

---

## 7. Usage in Pipeline

### 7.1 Main Processing Call

```python
# In part_rpf.py - simple_test()

### 2D pose estimation ###
track_kpts = []
if track_bboxes.shape[0] != 0:
    # Call pose estimator
    poses = self.pose_estimator.predict(
        img.squeeze().cpu(),      # Original image
        img_metas,                 # Image metadata
        track_bboxes[:, :4].cpu(), # Bounding boxes [x1, y1, x2, y2]
        track_bboxes[:, 4].cpu(),  # Bbox scores
        bbox_ids=track_ids,        # Track IDs
        rescale=rescale            # Whether to rescale coords
    )
    
    # Extract results
    track_bboxes = []
    track_ids = []
    for ps in poses:
        track_bboxes.append(torch.cat([ps['bbox'], ps['bbox_score'].unsqueeze(0)]).tolist())
        track_ids.append(ps['bbox_id'].tolist())
        track_kpts.append(torch.cat((ps['keypoints'], ps['kp_score']), axis=1).tolist())
```

### 7.2 How Features Are Used

| Feature | Used For |
|---------|----------|
| `keypoints` (x, y) | Orientation estimation input |
| `kp_score` (confidence) | Filtering unreliable joints |
| `joints_feature` | Part of classifier input (optional) |
| Joint positions | Determining body part visibility masks |
| Neck position | Body orientation calculation |

### 7.3 Orientation Estimation Input

```python
# In part_rpf.py
### orientation estimation ###
if track_bboxes.shape[0] != 0:
    # Process keypoints for HOE network
    _, processed_kpts = process_kpts(
        poses, 
        input_height=self.image_patch_size[1], 
        input_width=self.image_patch_size[0]
    )
    
    # Estimate orientation (0-360 degrees)
    hoe_outputs = self.orientation_estimator(processed_kpts)
    track_oris = hoe_outputs.argmax(axis=1) * 5  # Quantized to 5-degree bins
```

---

## 8. File References

| Component | File Path |
|-----------|-----------|
| Pose Estimator Loader | `mmtrack/models/pose/PoseEstimateLoader.py` |
| FastPose Model | `mmtrack/models/pose/SPPE/src/models/FastPose.py` |
| SE-ResNet Backbone | `mmtrack/models/pose/SPPE/src/models/layers/SE_Resnet.py` |
| DUC Layer | `mmtrack/models/pose/SPPE/src/models/layers/DUC.py` |
| Inference Wrapper | `mmtrack/models/pose/SPPE/src/main_fast_inference.py` |
| Prediction Utils | `mmtrack/models/pose/SPPE/src/utils/eval.py` |
| Pose NMS | `mmtrack/models/pose/pPose_nms.py` |
| Image Cropping | `mmtrack/models/pose/SPPE/src/utils/img.py` |
| Tracklet (Feature Storage) | `mmtrack/models/identifier/track_center/tracklet.py` |
| Pretrained Weights | `mmtrack/models/pose/Models/sppe/fast_res50_256x192.pth` |

---

## Appendix: Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `resnet50` | Backbone network (resnet50 or resnet101) |
| `input_height` | 224 | Input patch height |
| `input_width` | 160 | Input patch width |
| `scoreThreds` | 0.3 | Minimum keypoint confidence threshold |
| `matchThreds` | 5 | Max overlapping keypoints for NMS |
| `conf_thr` | 0.3 | Confidence threshold for joints_feature |
| `temp` | 0.1 | Temperature for softmax normalization |
