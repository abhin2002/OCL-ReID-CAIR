# OCL-ReID ByteTracker: Complete Frame Processing Flow Documentation

## Overview

This document provides a comprehensive breakdown of how each video frame is processed through the OCL-ReID ByteTracker system. The system combines **Object Detection**, **Multi-Object Tracking (ByteTrack)**, **Pose Estimation**, **Orientation Estimation**, **Re-Identification**, and **Online Continual Learning** for robust person following.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point: run_video.py](#2-entry-point-run_videopy)
3. [Frame-by-Frame Processing Flow](#3-frame-by-frame-processing-flow)
4. [Detailed Component Breakdown](#4-detailed-component-breakdown)
5. [State Machine for Target Identification](#5-state-machine-for-target-identification)
6. [Data Structures](#6-data-structures)
7. [Complete Flow Diagram](#7-complete-flow-diagram)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            OCL-ReID ByteTracker Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐ │
│  │  Video   │───▶│ Detection │───▶│ ByteTrack  │───▶│   Pose   │───▶│ Orient. │ │
│  │  Frame   │    │  (YOLOX)  │    │  Tracker   │    │  Estim.  │    │  Estim. │ │
│  └──────────┘    └───────────┘    └────────────┘    └──────────┘    └─────────┘ │
│                                          │                               │       │
│                                          ▼                               ▼       │
│                               ┌──────────────────────────────────────────┐       │
│                               │         Identifier (OCL-ReID)            │       │
│                               │  ┌─────────────┐  ┌──────────────────┐   │       │
│                               │  │   Feature   │  │  State Machine   │   │       │
│                               │  │  Extraction │  │  (Init→Train→    │   │       │
│                               │  │   (ResNet)  │  │   Track→ReID)    │   │       │
│                               │  └─────────────┘  └──────────────────┘   │       │
│                               │  ┌─────────────┐  ┌──────────────────┐   │       │
│                               │  │  Memory     │  │   Classifier     │   │       │
│                               │  │  Manager    │  │ (Ridge + CNN)    │   │       │
│                               │  └─────────────┘  └──────────────────┘   │       │
│                               └──────────────────────────────────────────┘       │
│                                          │                                       │
│                                          ▼                                       │
│                               ┌──────────────────┐                               │
│                               │   Target Result  │                               │
│                               │   (ID, BBox,     │                               │
│                               │    Confidence)   │                               │
│                               └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entry Point: run_video.py

### 2.1 Initialization

**File:** `run_video.py`

```python
class TargetIdentificationEvaluator:
    def __init__(self, hyper_config, config, identifier_config):
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | Config files | Load hyperparameters | `hyper_config`, `identifier_params` |
| 2 | Video path / Webcam ID | Determine input source | `self.input` |
| 3 | - | Initialize random seeds | Reproducible results |
| 4 | Configs | Create `Tracker` object | `self.tracker` |

### 2.2 Tracker Initialization (Evaluator.py)

**File:** `Evaluator.py`

```python
class Tracker:
    def init_tracker(self, hyper_config, seed):
        return init_model(self.config, self.checkpoint, device=self.device, 
                         hyper_config=hyper_config, identifier_config=self.identifier_params, seed=seed)
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | Config | Build complete model (YOLOX + ByteTracker + ReID + Identifier) | `model` (PartRPF) |
| 2 | Checkpoint | Load pretrained weights | Initialized model |
| 3 | Device | Move model to GPU | CUDA-enabled model |

---

## 3. Frame-by-Frame Processing Flow

### 3.1 Main Loop (run_video.py → run_video())

```python
while True:
    # 1. Get frame
    img = next(iter(imgs))  # or webcam.read()
    
    # 2. Initialize target (Frame 0 only)
    if frame_idx == 1:
        init_bbox = list(cv2.selectROI(...))  # or from gt_bbox_file
    
    # 3. Infer
    result_dict, raw_dict = self.tracker.infer(img, init_bbox, frame_idx - 1)
```

### 3.2 Tracker.infer() - Main Inference

**File:** `Evaluator.py`

```python
def infer(self, image_fname, target_gt_bbox, frame_id):
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | `image_fname`, `frame_id`, `gt_bbox` | Call `inference_rpf()` | `raw_result` dictionary |
| 2 | `raw_result` | Parse tracking results | `result` dict: `{id: [x1,y1,x2,y2]}` |
| 3 | `result`, `target_bbox` | Calculate distance/match | `return_result` with target info |

---

## 4. Detailed Component Breakdown

### 4.1 inference_rpf() - API Entry

**File:** `mmtrack/apis/inference.py`

```python
def inference_rpf(model, img, frame_id, gt_bbox):
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | `img` (path or ndarray) | Prepare data dict | `data = {img, img_info, img_prefix}` |
| 2 | `data` | Apply test pipeline (resize, normalize) | Preprocessed `data` |
| 3 | `data` | Collate and scatter to GPU | Batched tensor data |
| 4 | `data`, `gt_bbox` | Forward through model | `result` dictionary |

**Test Pipeline Transformations:**
```
LoadImageFromFile → Resize → Normalize → Pad → Collect
```

---

### 4.2 PartRPF.simple_test() - Core Processing

**File:** `mmtrack/models/rpf/part_rpf.py`

This is the **main processing function** where all components work together.

#### Phase 1: Object Detection (YOLOX)

```python
det_results = self.detector.simple_test(img, img_metas, rescale=rescale)
```

| Input | Operation | Output |
|-------|-----------|--------|
| `img`: Tensor (N, C, H, W) | YOLOX forward pass | `det_results`: List of bbox arrays |
| `img_metas`: List[dict] | NMS filtering | `bbox_results`: (num_classes, num_dets, 5) |

**Detection Output Format:**
```python
det_bboxes = [[x1, y1, x2, y2, score], ...]  # Shape: (N, 5)
det_labels = [0, 0, 0, ...]  # All zeros (person class)
```

---

#### Phase 2: ByteTrack Multi-Object Tracking

**File:** `mmtrack/models/trackers/byte_tracker.py`

```python
track_bboxes, track_labels, track_ids = self.tracker.track(
    img=img, img_metas=img_metas, model=self,
    bboxes=det_bboxes, labels=det_labels, frame_id=frame_id, rescale=rescale)
```

**ByteTrack Algorithm Steps:**

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| **1. Kalman Predict** | Previous track states | Predict current positions | Updated `mean`, `covariance` |
| **2. First Match (High Conf)** | Detections with score > 0.6 | IOU-based Hungarian matching | Matched track IDs |
| **3. Tentative Match** | Unmatched high-conf dets | Match with unconfirmed tracks | Updated tentative tracks |
| **4. Second Match (Low Conf)** | Detections with 0.1 < score < 0.6 | Match with lost confirmed tracks | Recovered tracks |
| **5. Initialize New Tracks** | Unmatched detections | Create new track entries | New track IDs |
| **6. Update Tracks** | All matched pairs | Kalman Filter update | Updated track states |

**Key Data Structures:**

```python
self.tracks = {
    track_id: {
        'bboxes': [...],      # History of bboxes
        'frame_ids': [...],   # Frame IDs when detected
        'mean': np.array(...),      # Kalman state [cx, cy, a, h, vx, vy, va, vh]
        'covariance': np.array(...),  # Kalman covariance
        'tentative': bool,    # Confirmed or tentative track
        'labels': [...],      # Class labels
    }
}
```

**Kalman Filter State:**
```
State vector: [center_x, center_y, aspect_ratio, height, v_cx, v_cy, v_a, v_h]
Measurement: [center_x, center_y, aspect_ratio, height]
```

---

#### Phase 3: 2D Pose Estimation

```python
poses = self.pose_estimator.predict(img, img_metas, track_bboxes[:, :4], 
                                     track_bboxes[:, 4], bbox_ids=track_ids, rescale=rescale)
```

| Input | Operation | Output |
|-------|-----------|--------|
| `img`: Original image | Crop person patches | Cropped images |
| `track_bboxes`: (N, 5) | SPPE FastPose inference | Keypoint predictions |
| `bbox_ids`: Track IDs | Map keypoints to tracks | Pose per track |

**Keypoint Output (14 joints + Neck):**
```python
track_kpts = [
    [[x1, y1, conf1], [x2, y2, conf2], ...],  # 14 keypoints per person
    ...
]
# Joints: [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, 
#          LHip, RHip, LKnee, RKnee, LAnkle, RAnkle, Neck]
```

---

#### Phase 4: Orientation Estimation

```python
_, processed_kpts = process_kpts(poses, input_height=self.image_patch_size[1], 
                                  input_width=self.image_patch_size[0])
hoe_outputs = self.orientation_estimator(processed_kpts)
track_oris = hoe_outputs.argmax(axis=1) * 5  # Degree (0-355°)
```

| Input | Operation | Output |
|-------|-----------|--------|
| `poses`: Keypoint data | Normalize keypoints | `processed_kpts` |
| `processed_kpts` | HOE network inference | Orientation logits |
| Logits | argmax × 5 | Orientation in degrees |

**Orientation Used For:**
- Determining front/back view (binary: 0 = Front, 1 = Back)
- Selecting which part features to use for identification

---

#### Phase 5: Target Initialization (Frame 0)

```python
if len(track_bboxes) != 0 and self.target_id is None:
    self.init(gt_bbox, raw_result)
    if self.target_id is not None:
        self.identifier.init_identifier(target_id=self.target_id, rpf_model=self)
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | `gt_bbox`, all tracks | Compute IOU between GT and tracks | IOU scores |
| 2 | IOU scores | Find track with max IOU > threshold | `target_id` |
| 3 | `target_id`, `rpf_model` | Initialize identifier and classifier | Ready for tracking |

---

#### Phase 6: Target Identification (OCL-ReID)

**File:** `mmtrack/models/identifier/oclreid_identifier.py`

```python
ident_result = self.identifier.identify(
    img=img, img_metas=img_metas, model=self,
    tracks=result, frame_id=frame_id, rescale=rescale, gt_bbox=gt_bbox)
```

##### Step 6.1: Create Tracklets

```python
for id in tracks.keys():
    self.tracklets[id] = Tracklet(img=img, img_metas=img_metas,
                                   observation=tracks[id], rescale=rescale,
                                   img_scale=(self.img_patch_height, self.img_patch_width))
```

**Tracklet Data Structure:**
```python
class Tracklet:
    bbox: List[4]           # [x1, y1, x2, y2]
    image_patch: Tensor     # Cropped and resized (1, 3, H, W)
    bbox_feature: Tuple     # (normalized_height, normalized_width)
    kpts: np.array          # 14 keypoints
    joints_feature: Tensor  # Softmax-weighted joint positions
    deep_feature: Tensor    # CNN features (will be filled)
    visibility_score: Tensor
    target_confidence: float
```

##### Step 6.2: Extract Deep Features

```python
self.extract_features(tracklets=self.tracklets, model=model.reid)
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | Image patches | Stack into batch | `img_patches`: (N, 3, H, W) |
| 2 | `img_patches` | Normalize (mean/std) | Normalized tensor |
| 3 | Normalized tensor | ResNet18 backbone | Feature maps (N, 512, 8, 4) |
| 4 | Feature maps | Part-based pooling | Part features (N, 5, 512) or (N, 10, 512) |

**Feature Extraction Details:**
```python
# ResNet18 Backbone
x = self.clf.extract_feat(imgs, stage="backbone")  # (B, 512, 8, 4)

# Part-based Feature Pooling (with visibility-aware masking)
# Parts: [HEAD, TORSO, LEGS, FEET, GLOBAL] × [FRONT, BACK] = 10 parts
```

##### Step 6.3: Update State Machine

```python
next_state = self.state.update(identifier=self, tracklets=self.tracklets)
```

See [Section 5: State Machine](#5-state-machine-for-target-identification) for details.

---

### 4.3 Classifier Operations

**File:** `mmtrack/models/identifier/classifier/part_ocl_weighted_classifier.py`

#### Prediction (predict())

```python
def predict(self, tracklets: dict, state="tracking"):
```

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | `tracklet.deep_feature` | Get part features (5 or 10 parts) | Feature vector |
| 2 | `tracklet.visibility_indicator` | Determine visible parts | Binary mask |
| 3 | Visible features | Ridge Regression predict | Part scores |
| 4 | Part scores | Weighted average | `target_confidence` |

**Score Calculation:**
```python
# For each visible part
part_scores[part_idx] = self.st_clfs[part_idx].predict(feature[[i]])

# Aggregate
if len(part_scores) != 0:
    avg_score = sum(part_scores.values()) / len(part_scores)
else:
    avg_score = 0.5  # Maximum entropy
```

#### Training (train())

```python
def train(self):
    # Short-term classifier (Ridge Regression)
    stf_x, stf_y = self.memory_manager.retrieve_st_features()
    self.train_st(stf_x, stf_y)
    
    # Long-term classifier (CNN fine-tuning)
    st_x, st_y, vis_map, vis_indicator = self.memory_manager.retrieve_st()
    self.train_lt(st_x, st_y, vis_map, vis_indicator)
```

| Component | Input | Operation | Output |
|-----------|-------|-----------|--------|
| **Short-term** | ST memory features | Ridge Regression fit | Updated `st_clfs` |
| **Long-term** | LT memory images | CNN forward + backward | Updated backbone |

**Loss Functions:**
- CrossEntropyLoss for global features
- PartAveragedTripletLoss for part features

---

### 4.4 Memory Manager

**File:** `mmtrack/models/identifier/memory_manager/part_ocl_memory_manager.py`

```python
def update(self, tracklets: dict, target_id: int):
```

| Memory Type | Purpose | Update Strategy |
|-------------|---------|-----------------|
| **Short-term (ST)** | Recent samples | FIFO / Reservoir sampling |
| **Long-term (LT)** | Keyframe storage | Selective based on loss |

**Memory Structure:**
```python
self.memory = {
    "st_set": {
        "buffer_img": Tensor,        # Image patches
        "buffer_label": Tensor,      # 0 (negative) or 1 (positive)
        "buffer_feature_0": Tensor,  # Part 0 features
        "buffer_feature_1": Tensor,  # Part 1 features
        ...
        "buffer_tracker": ClassTracker  # Tracks class distribution
    },
    "lt_set": { ... }  # Same structure
}
```

---

## 5. State Machine for Target Identification

**File:** `mmtrack/models/identifier/states/`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           State Machine Flow                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│    ┌─────────────┐                                                            │
│    │  INITIAL    │◄───────────────────────────────────────────────────────┐  │
│    │   STATE     │  Target lost during initial training                   │  │
│    └──────┬──────┘                                                        │  │
│           │ Target selected (IOU match with GT)                           │  │
│           ▼                                                               │  │
│    ┌─────────────────┐                                                    │  │
│    │    INITIAL      │                                                    │  │
│    │   TRAINING      │◀────────────────────────────────────┐              │  │
│    │     STATE       │                                     │              │  │
│    └────────┬────────┘                                     │              │  │
│             │ Collected enough positive samples            │              │  │
│             │ (num_samples >= initial_training_num_samples)│              │  │
│             ▼                                              │              │  │
│    ┌─────────────────┐                                     │              │  │
│    │    TRACKING     │◀─────────────────────────┐          │              │  │
│    │     STATE       │                          │          │              │  │
│    └────────┬────────┘                          │          │              │  │
│             │                                   │          │              │  │
│             │ Target not found OR               │          │              │  │
│             │ confidence < id_switch_thresh     │          │              │  │
│             ▼                                   │          │              │  │
│    ┌─────────────────┐                          │          │              │  │
│    │     REID        │                          │          │              │  │
│    │     STATE       │──────────────────────────┘          │              │  │
│    └────────┬────────┘  Re-identified target               │              │  │
│             │           (consecutive positive count)        │              │  │
│             │                                              │              │  │
│             └──────────────────────────────────────────────┘              │  │
│               Target found in first frame                                  │  │
│                                                                            │  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### State Descriptions

| State | Purpose | Entry Condition | Exit Condition |
|-------|---------|-----------------|----------------|
| **InitialState** | Wait for target selection | Start | Target matched with GT |
| **InitialTrainingState** | Collect positive samples | Target selected | Enough samples collected |
| **TrackingState** | Normal tracking | Training complete | Target lost or ID switch |
| **ReidState** | Re-identify target | Target lost | Target re-identified |

### State Operations

#### InitialTrainingState.update()
```python
1. Check if target_id in tracklets
2. Update memory with target sample (positive)
3. Update classifier (train)
4. Increment sample counter
5. If samples >= threshold → TrackingState
```

#### TrackingState.update()
```python
1. Predict confidence for all tracklets
2. If target not in tracklets → ReidState
3. If confidence < id_switch_thresh → ReidState (ID switch detected)
4. If confidence < min_threshold → Don't update (target occluded)
5. Else → Update memory and classifier
```

#### ReidState.update()
```python
1. Predict confidence for all tracklets
2. For each tracklet with confidence > reid_thresh:
   - Increment positive_count[id]
   - If positive_count >= consecutive_threshold:
     → TrackingState(new_target_id)
```

---

## 6. Data Structures

### 6.1 Raw Result Dictionary

```python
raw_result = {
    # Detection
    'det_bboxes': List[np.array],     # [(N, 5)] - [x1,y1,x2,y2,score]
    
    # Tracking
    'track_bboxes': List[np.array],   # [(M, 5)]
    'track_kpts': List[List],         # [[14×3], ...] - keypoints
    'track_oris': List[int],          # [ori1, ori2, ...] - degrees
    
    # Ground Truth
    'gt_bbox': np.array,              # [x1, y1, x2, y2]
    
    # Identification (added by identifier)
    'target_id': int,                 # Current target track ID
    'target_bbox': List[4],           # Target bounding box
    'target_conf': float,             # Target confidence score
    'state': str,                     # Current state name
    'threshold': float,               # ReID threshold
    'tracks_target_conf_bbox': {      # All tracks info
        track_id: [_, confidence, bbox]
    }
}
```

### 6.2 Tracklet Object

```python
Tracklet:
    # Bounding Box Info
    bbox: List[4]                 # [x1, y1, x2, y2]
    bbox_score: float             # Detection confidence
    bbox_feature: Tuple[2]        # (norm_height, norm_width)
    
    # Image Patch
    image_patch: Tensor           # (1, 3, 256, 192)
    img_scale: Tuple[2]           # (height, width)
    
    # Keypoints
    kpts: np.array                # (14, 3) - [x, y, conf]
    joints_feature: Tensor        # Softmax-weighted positions
    
    # Deep Features (filled by identifier)
    deep_feature: Tensor          # (5, 512) or (10, 512) part features
    visibility_score: Tensor      # Part visibility scores
    visibility_indicator: Tensor  # Binary part visibility
    
    # Identification
    target_confidence: float      # Probability of being target
    part_target_confidence: dict  # {part_idx: confidence}
    binary_ori: int               # 0=Front, 1=Back
```

### 6.3 Track Object (ByteTracker)

```python
Track:
    bboxes: List[Tensor]          # History of bboxes
    labels: List[int]             # Class labels
    frame_ids: List[int]          # Frame numbers
    
    # Kalman Filter State
    mean: np.array(8)             # [cx, cy, a, h, vx, vy, va, vh]
    covariance: np.array(8,8)     # State covariance
    
    # Status
    tentative: bool               # Confirmed or tentative
```

---

## 7. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE FRAME PROCESSING FLOW                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FRAME INPUT                                                                     │
│  ═══════════                                                                     │
│  img: BGR image (H, W, 3)                                                        │
│  frame_id: int                                                                   │
│  gt_bbox (frame 0): [x1, y1, x2, y2]                                            │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: DATA PREPROCESSING                                               │   │
│  │ File: mmtrack/apis/inference.py                                          │   │
│  │                                                                          │   │
│  │ Input:  img (str or ndarray)                                             │   │
│  │ Operations:                                                              │   │
│  │   • LoadImageFromFile/Webcam                                             │   │
│  │   • Resize to (800, 1440)                                                │   │
│  │   • Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]│  │
│  │   • Pad to multiple of 32                                                │   │
│  │   • ToTensor, Collect                                                    │   │
│  │ Output: data dict {img: (1,3,H,W), img_metas: [...]}                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: OBJECT DETECTION (YOLOX-L)                                        │   │
│  │ File: mmtrack/models/rpf/part_rpf.py → simple_test()                     │   │
│  │                                                                          │   │
│  │ Input:  img tensor (1, 3, H, W)                                          │   │
│  │ Operations:                                                              │   │
│  │   • YOLOX backbone (CSPDarknet)                                          │   │
│  │   • YOLOX neck (PAFPN)                                                   │   │
│  │   • YOLOX head (decoupled head)                                          │   │
│  │   • NMS (score_thr=0.6, iou_thr=0.35)                                    │   │
│  │ Output:                                                                  │   │
│  │   det_bboxes: Tensor (N, 5) - [x1, y1, x2, y2, score]                    │   │
│  │   det_labels: Tensor (N,) - all zeros (person class)                     │   │
│  │ Time: ~15-25ms                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: MULTI-OBJECT TRACKING (ByteTrack)                                 │   │
│  │ File: mmtrack/models/trackers/byte_tracker.py → track()                  │   │
│  │                                                                          │   │
│  │ Input:                                                                   │   │
│  │   • det_bboxes: (N, 5)                                                   │   │
│  │   • Previous frame tracks                                                │   │
│  │                                                                          │   │
│  │ Operations:                                                              │   │
│  │   3.1 Kalman Filter Prediction                                           │   │
│  │       • Predict new positions from previous states                       │   │
│  │       • mean' = F @ mean                                                 │   │
│  │       • cov' = F @ cov @ F.T + Q                                         │   │
│  │                                                                          │   │
│  │   3.2 First Association (High Confidence)                                │   │
│  │       • Filter detections with score > 0.6                               │   │
│  │       • Compute IOU matrix between tracks and detections                 │   │
│  │       • Weight IOU by detection scores                                   │   │
│  │       • Hungarian algorithm (LAP) for matching                           │   │
│  │       • Match threshold: IOU > 0.1                                       │   │
│  │                                                                          │   │
│  │   3.3 Tentative Track Matching                                           │   │
│  │       • Match unmatched high-conf dets with unconfirmed tracks           │   │
│  │       • Match threshold: IOU > 0.3                                       │   │
│  │                                                                          │   │
│  │   3.4 Second Association (Low Confidence)                                │   │
│  │       • Filter detections with 0.1 < score < 0.6                         │   │
│  │       • Match with unmatched confirmed tracks (from step 3.2)            │   │
│  │       • Match threshold: IOU > 0.5                                       │   │
│  │                                                                          │   │
│  │   3.5 Initialize New Tracks                                              │   │
│  │       • Create new tracks for unmatched detections with score > 0.7      │   │
│  │       • Initialize Kalman state with bbox                                │   │
│  │                                                                          │   │
│  │   3.6 Kalman Filter Update                                               │   │
│  │       • Update matched tracks with new observations                      │   │
│  │       • Remove tracks not seen for 10 frames                             │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   track_bboxes: Tensor (M, 5) - tracked person bboxes                    │   │
│  │   track_ids: Tensor (M,) - unique track IDs                              │   │
│  │   track_labels: Tensor (M,) - class labels                               │   │
│  │ Time: ~2-5ms                                                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: 2D POSE ESTIMATION (AlphaPose/SPPE)                               │   │
│  │ File: mmtrack/models/pose/PoseEstimateLoader.py                          │   │
│  │                                                                          │   │
│  │ Input:                                                                   │   │
│  │   • Original image                                                       │   │
│  │   • track_bboxes: (M, 5)                                                 │   │
│  │                                                                          │   │
│  │ Operations:                                                              │   │
│  │   • Crop person patches from image using bboxes                          │   │
│  │   • Resize patches to (160, 224)                                         │   │
│  │   • ResNet50 backbone + deconv layers                                    │   │
│  │   • Heatmap regression for 13 keypoints                                  │   │
│  │   • Add neck keypoint (average of shoulders)                             │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   track_kpts: List[M] of (14, 3) arrays                                  │   │
│  │   Each keypoint: [x, y, confidence]                                      │   │
│  │   Keypoints: Nose, LShoulder, RShoulder, LElbow, RElbow,                 │   │
│  │              LWrist, RWrist, LHip, RHip, LKnee, RKnee,                    │   │
│  │              LAnkle, RAnkle, Neck                                        │   │
│  │ Time: ~10-20ms                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: ORIENTATION ESTIMATION (HOE)                                      │   │
│  │ File: mmtrack/models/orientation/                                        │   │
│  │                                                                          │   │
│  │ Input:                                                                   │   │
│  │   • track_kpts: (M, 14, 3) - keypoints with confidence                   │   │
│  │                                                                          │   │
│  │ Operations:                                                              │   │
│  │   • Normalize keypoints to [0, 1]                                        │   │
│  │   • Process through HOE network                                          │   │
│  │   • Output 72 orientation bins (360°/5° = 72)                            │   │
│  │   • Select max bin × 5 = orientation in degrees                          │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   track_oris: List[M] of int (0-355 degrees)                             │   │
│  │   binary_ori: 0 (Front/0-90°, 270-360°) or 1 (Back/90-270°)              │   │
│  │ Time: ~2-5ms                                                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: TARGET INITIALIZATION (Frame 0 Only)                              │   │
│  │ File: mmtrack/models/rpf/part_rpf.py → init()                            │   │
│  │                                                                          │   │
│  │ Input:                                                                   │   │
│  │   • gt_bbox: [x1, y1, x2, y2] (from user selection or file)              │   │
│  │   • All tracked persons                                                  │   │
│  │                                                                          │   │
│  │ Operations:                                                              │   │
│  │   • Compute IOU between GT bbox and each track bbox                      │   │
│  │   • Find track with maximum IOU > threshold (0.4)                        │   │
│  │   • Set that track ID as target_id                                       │   │
│  │   • Initialize identifier with target_id                                 │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   target_id: int (the track ID of target person)                         │   │
│  │   target_bbox: [x1, y1, x2, y2]                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 7: TARGET IDENTIFICATION (OCL-ReID)                                  │   │
│  │ File: mmtrack/models/identifier/oclreid_identifier.py                    │   │
│  │                                                                          │   │
│  │ Input:                                                                   │   │
│  │   • img: (1, 3, H, W)                                                    │   │
│  │   • tracks: {id: [x1,y1,x2,y2,score,kpts,ori], ...}                      │   │
│  │   • target_id: int                                                       │   │
│  │                                                                          │   │
│  │ STEP 7.1: CREATE TRACKLETS                                               │   │
│  │ ─────────────────────────────────────────────                            │   │
│  │   • For each track, create Tracklet object                               │   │
│  │   • Crop image patch: (1, 3, 256, 192)                                   │   │
│  │   • Store bbox, keypoints, orientation                                   │   │
│  │                                                                          │   │
│  │ STEP 7.2: EXTRACT DEEP FEATURES                                          │   │
│  │ ─────────────────────────────────────────────                            │   │
│  │   • Stack all image patches: (M, 3, 256, 192)                            │   │
│  │   • Normalize: (patch - mean) / std                                      │   │
│  │   • ResNet18 backbone forward                                            │   │
│  │   • Output: feature maps (M, 512, 8, 4)                                  │   │
│  │                                                                          │   │
│  │ STEP 7.3: PART-BASED FEATURE POOLING                                     │   │
│  │ ─────────────────────────────────────────────                            │   │
│  │   • Divide feature map into 4 horizontal strips (HEAD, TORSO, LEGS, FEET)│   │
│  │   • Apply visibility-aware masking based on keypoints                    │   │
│  │   • Global average pooling per part: (M, 5, 512)                         │   │
│  │   • If using orientation: separate front/back → (M, 10, 512)             │   │
│  │                                                                          │   │
│  │ STEP 7.4: STATE MACHINE UPDATE                                           │   │
│  │ ─────────────────────────────────────────────                            │   │
│  │   See State Machine section for details                                  │   │
│  │   States: Initial → InitialTraining → Tracking ↔ ReID                   │   │
│  │                                                                          │   │
│  │ STEP 7.5: PREDICT TARGET CONFIDENCE                                      │   │
│  │ ─────────────────────────────────────────────                            │   │
│  │   • For each tracklet:                                                   │   │
│  │     - Get part features based on orientation                             │   │
│  │     - For each visible part:                                             │   │
│  │       · Ridge Regression predict: score = w @ feature + b                │   │
│  │     - Average visible part scores → target_confidence                    │   │
│  │   • Identify target based on confidence and state                        │   │
│  │                                                                          │   │
│  │ STEP 7.6: UPDATE MEMORY & CLASSIFIER (in Tracking State)                 │   │
│  │ ─────────────────────────────────────────────────────────                │   │
│  │   Memory Update:                                                         │   │
│  │     • Add target patch as positive sample                                │   │
│  │     • Add other tracks as negative samples                               │   │
│  │     • Short-term: FIFO buffer (recent frames)                            │   │
│  │     • Long-term: Reservoir sampling (keyframes)                          │   │
│  │                                                                          │   │
│  │   Classifier Training:                                                   │   │
│  │     • Short-term (Ridge Regression):                                     │   │
│  │       - Retrieve features from ST memory                                 │   │
│  │       - Fit Ridge: (X, y) for each part                                  │   │
│  │     • Long-term (CNN fine-tuning):                                       │   │
│  │       - Retrieve images from LT memory                                   │   │
│  │       - Forward + compute loss (CE + Triplet)                            │   │
│  │       - Backward + optimizer step                                        │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   ident_result = {                                                       │   │
│  │     'state': str,                # Current state                         │   │
│  │     'target_id': int,            # Identified target                     │   │
│  │     'target_conf': float,        # Target confidence                     │   │
│  │     'threshold': float,          # ReID threshold                        │   │
│  │     'tracks_target_conf_bbox': { # All tracks info                       │   │
│  │       id: [_, conf, bbox], ...                                           │   │
│  │     }                                                                    │   │
│  │   }                                                                      │   │
│  │ Time: ~15-30ms                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 8: RESULT AGGREGATION & OUTPUT                                       │   │
│  │ File: run_video.py                                                       │   │
│  │                                                                          │   │
│  │ Input: raw_result dict from model                                        │   │
│  │                                                                          │   │
│  │ Operations:                                                              │   │
│  │   • Extract target_id, target_conf, target_bbox                          │   │
│  │   • Store results in JSON (if output_json specified)                     │   │
│  │   • Visualize results on frame (if show_result or output specified)      │   │
│  │   • Draw bounding boxes (green=target, blue=others)                      │   │
│  │   • Display confidence scores                                            │   │
│  │                                                                          │   │
│  │ Output:                                                                  │   │
│  │   • Displayed/saved frame with annotations                               │   │
│  │   • JSON entry: {                                                        │   │
│  │       'target_info': [id, x1, y1, x2, y2, conf],                        │   │
│  │       'det_bboxes': [...],                                               │   │
│  │       'tracks_target_conf_bbox': {...}                                   │   │
│  │     }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  TOTAL PROCESSING TIME PER FRAME: ~50-100ms (10-20 FPS)                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `select_target_threshold` | 0.4 | IOU threshold for initial target selection |
| `obj_score_thrs.high` | 0.6 | Detection score for first association |
| `obj_score_thrs.low` | 0.1 | Detection score for second association |
| `init_track_thr` | 0.7 | Score threshold to initialize new track |
| `num_frames_retain` | 10 | Frames to keep lost track |
| `reid_pos_confidence_thresh` | - | Threshold for positive identification |
| `id_switch_detection_thresh` | - | Threshold to detect ID switch |
| `initial_training_num_samples` | - | Samples needed before tracking state |
| `reid_positive_count` | - | Consecutive positives for re-identification |

---

## Appendix B: File Reference

| Component | File Path |
|-----------|-----------|
| Main Entry | `run_video.py` |
| Tracker Wrapper | `Evaluator.py` |
| API Functions | `mmtrack/apis/inference.py` |
| RPF Model | `mmtrack/models/rpf/part_rpf.py` |
| ByteTracker | `mmtrack/models/trackers/byte_tracker.py` |
| Identifier | `mmtrack/models/identifier/oclreid_identifier.py` |
| Classifier | `mmtrack/models/identifier/classifier/part_ocl_weighted_classifier.py` |
| States | `mmtrack/models/identifier/states/` |
| Tracklet | `mmtrack/models/identifier/track_center/tracklet.py` |
| ReID Model | `mmtrack/models/reid/part_weighted_classifier.py` |
| Memory Manager | `mmtrack/models/identifier/memory_manager/` |
| Config Files | `tpt_configs/`, `configs/rpf/` |

---

## Appendix C: Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| OCL | Online Continual Learning |
| ReID | Re-Identification |
| RPF | Robot Person Following |
| MOT | Multi-Object Tracking |
| IOU | Intersection over Union |
| YOLOX | You Only Look Once X |
| HOE | Human Orientation Estimation |
| SPPE | Single Person Pose Estimation |
| ST | Short-Term |
| LT | Long-Term |
| LAP | Linear Assignment Problem |
